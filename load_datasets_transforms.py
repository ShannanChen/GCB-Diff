from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
from monai import transforms, data

from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    AsChannelFirstd,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    ResizeWithPadOrCropd,
    EnsureChannelFirst,
    Resized,
    LoadImaged,
    RandSpatialCropSamplesd,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)

import numpy as np
from collections import OrderedDict
import glob

def train_data_loader(data_dir):
    root_dir = data_dir
    train_samples = {}
    valid_samples = {}

    ## Input training data
    train_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTr', '*.nii.gz')))
    train_img_T1 = sorted(glob.glob(os.path.join(root_dir, 'imagesTr_T1', '*.nii.gz')))
    train_img_T2 = sorted(glob.glob(os.path.join(root_dir, 'imagesTr_T2', '*.nii.gz')))
    train_img_flair = sorted(glob.glob(os.path.join(root_dir, 'imagesTr_flair', '*.nii.gz')))
    train_label = sorted(glob.glob(os.path.join(root_dir, 'labelsTr', '*.nii.gz')))
    train_samples['images'] = train_img
    train_samples['images_T1'] = train_img_T1
    train_samples['images_T2'] = train_img_T2
    train_samples['images_flair'] = train_img_flair
    train_samples['labels'] = train_label

    ## Input validation data
    valid_img = sorted(glob.glob(os.path.join(root_dir, 'imagesVal', '*.nii.gz')))
    valid_img_T1 = sorted(glob.glob(os.path.join(root_dir, 'imagesVal_T1', '*.nii.gz')))
    valid_img_T2 = sorted(glob.glob(os.path.join(root_dir, 'imagesVal_T2', '*.nii.gz')))
    valid_img_flair = sorted(glob.glob(os.path.join(root_dir, 'imagesVal_flair', '*.nii.gz')))
    valid_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
    valid_samples['images'] = valid_img
    valid_samples['images_T1'] = valid_img_T1
    valid_samples['images_T2'] = valid_img_T2
    valid_samples['images_flair'] = valid_img_flair
    valid_samples['labels'] = valid_label

    return train_samples, valid_samples, 1


def test_data_loader(data_dir):
    root_dir = data_dir
    test_samples = {}

    ## Input inference data
    test_img = sorted(glob.glob(os.path.join(root_dir, 'imagesTs', '*.nii.gz')))
    test_img_T1 = sorted(glob.glob(os.path.join(root_dir, 'imagesTs_T1', '*.nii.gz')))
    test_img_T2 = sorted(glob.glob(os.path.join(root_dir, 'imagesTs_T2', '*.nii.gz')))
    test_img_flair = sorted(glob.glob(os.path.join(root_dir, 'imagesTs_flair', '*.nii.gz')))
    test_label = sorted(glob.glob(os.path.join(root_dir, 'labelsVal', '*.nii.gz')))
    test_samples['images'] = test_img
    test_samples['images_T1'] = test_img_T1
    test_samples['images_T2'] = test_img_T2
    test_samples['images_flair'] = test_img_flair
    test_samples['labels'] = test_label

    return test_samples,  1


def data_transforms(mode):
    if mode == 'train':
        crop_samples = 1
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "images_T1", "images_T2", "images_flair", "label"]),
                AddChanneld(keys=["image", "images_T1", "images_T2", "images_flair", "label"]),
                Orientationd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], axcodes="RAS"),
                transforms.CropForegroundd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], source_key="image"),
                transforms.RandSpatialCropd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], roi_size=[96, 96, 96], random_size=False),
                transforms.SpatialPadd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], spatial_size=(96, 96, 96)),
                transforms.RandFlipd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], prob=0.5, spatial_axis=2),
                transforms.NormalizeIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], nonzero=True, channel_wise=True),
                transforms.RandScaleIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], factors=0.1, prob=1),
                transforms.RandShiftIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], offsets=0.1, prob=1),
                transforms.ToTensord(keys=["image", "images_T1", "images_T2", "images_flair", "label"], ),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "images_T1", "images_T2", "images_flair", "label"]),
                AddChanneld(keys=["image", "images_T1", "images_T2", "images_flair", "label"]),
                Orientationd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], axcodes="RAS"),
                transforms.CropForegroundd(keys=["image",  "images_T1", "images_T2", "images_flair", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image",  "images_T1", "images_T2", "images_flair", "label"]),

            ]
        )
    else:
        crop_samples = None
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "images_T1", "images_T2", "images_flair"]),
                AddChanneld(keys=["image", "images_T1", "images_T2", "images_flair"]),
                Orientationd(keys=["image", "images_T1", "images_T2", "images_flair"], axcodes="RAS"),
                transforms.CropForegroundd(keys=["image", "images_T1", "images_T2", "images_flair", "label"], source_key="image"),
                transforms.NormalizeIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], nonzero=True, channel_wise=True),
                transforms.ToTensord(keys=["image", "images_T1", "images_T2", "images_flair", "label"]),
            ]
        )

    if mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms


def infer_post_transforms(path, test_transforms):

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            # orig_keys="image, image_adc, image_flair",  # get the previously applied pre_transforms information on the `img` data field,
            orig_keys="images, images_T1, images_T2, images_flair",  # get the previously applied pre_transforms information on the `img` data field,
            # then invert `pred` based on this information. we can use same info
            # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
            # for example, may need the `affine` to invert `Spacingd` transform,
            # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
            # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
            # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        ## If monai version <= 0.6.0:
        # AsDiscreted(keys="pred", argmax=True, n_classes=1),
        ## If moani version > 0.6.0:
        AsDiscreted(keys="pred", argmax=True),
        # KeepLargestConnectedComponentd(keys='pred', applied_labels=[1, 3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=path,
                   output_postfix="seg", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms



