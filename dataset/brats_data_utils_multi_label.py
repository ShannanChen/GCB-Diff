# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证
from ISLES2015.load_datasets_transforms import train_data_loader, test_data_loader, data_transforms, \
    infer_post_transforms
from monai.data import CacheDataset, DataLoader, decollate_batch

import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import Dataset
from monai.utils import set_determinism


def resample_img(
        image: sitk.Image,
        out_spacing=(2.0, 2.0, 2.0),
        out_size=None,
        is_label: bool = False,
        pad_value=0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):

        # file_identifizer = data_path.split("/")[-1].split("_")[-1]
        file_identifizer = data_path.split("/")[-1]
        image_paths = [
            os.path.join(data_path, file_identifizer + f"_adc.nii.gz"),
            os.path.join(data_path, file_identifizer + f"_flair.nii.gz"),
            os.path.join(data_path, file_identifizer + f"_dwi.nii.gz"),
        ]
        seg_path = os.path.join(data_path, file_identifizer + f"_msk.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))

        image_data = torch.tensor(np.array(image_data).astype(np.float32))
        image_data = dict(enumerate(image_data.flatten(), 1))
        seg_data = torch.tensor(np.expand_dims(np.array(seg_data).astype(np.int32), axis=0))
        seg_data = dict(enumerate(seg_data.flatten(), 1))
        return {
            "image": image_data,
            "label": seg_data
        }

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"数据读取出现问题，{self.datalist[i]}\n")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.datalist)


def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)  ## kfold为KFolf类的一个对象
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res


class Args:
    def __init__(self) -> None:
        self.workers = 8
        self.fold = 0
        self.batch_size = 2


def get_loader_brats(data_dir, batch_size=1, fold=0, num_workers=8):
    train_samples, valid_samples, out_classes = train_data_loader(data_dir)

    train_files = [
        {"image": image_name, "images_T1": images_T1, "images_T2": images_T2, "images_flair": images_flair, "label": label_name}
        for image_name, images_T1, images_T2, images_flair, label_name in
        zip(train_samples['images'], train_samples['images_T1'], train_samples['images_T2'], train_samples['images_flair'],
            train_samples['labels'])
    ]

    val_files = [
        {"image": image_name, "images_T1": images_T1, "images_T2": images_T2, "images_flair": images_flair, "label": label_name}
        for image_name, images_T1, images_T2, images_flair, label_name in
        zip(valid_samples['images'], valid_samples['images_T1'], valid_samples['images_T2'], valid_samples['images_flair'],
            valid_samples['labels'])
    ]

    set_determinism(seed=0)
    train_transforms, val_transforms = data_transforms(mode='train')

    ## Train Pytorch Data Loader and Caching
    print('Start caching datasets!')
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    ## Valid Pytorch Data Loader and Caching
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

    test_samples, out_classes = test_data_loader(data_dir)
    test_files = [
        {"image": image_name, "images_T1": images_T1, "images_T2": images_T2, "images_flair": images_flair, "label": label_name}
        for image_name, images_T1, images_T2, images_flair, label_name in
        zip(test_samples['images'], test_samples['images_T1'], test_samples['images_T2'], test_samples['images_flair'],
            test_samples['labels'])
    ]
    test_transforms = data_transforms(mode='test')

    ## Inference Pytorch Data Loader and Caching
    test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=0, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    loader = [train_loader, val_loader, test_loader]

    return loader
