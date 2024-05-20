import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer_old import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
import argparse
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from BraTS2020.networks.SwinUNETR.swin_unetr import SwinUNETR
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet
from load_datasets_transforms import test_data_loader, data_transforms, infer_post_transforms
from monai import transforms, data
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activationsd, Activations, Compose
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss

set_determinism(123)
import os

data_dir = "./datasets/MICCAI_2015/"
save_logdir = "./logs_brats/diffusion_seg_sigmoid_all_loss_embed/nii"

max_epoch = 300
batch_size = 1
val_every = 10
device = "cuda:0"

number_modality = 4
number_targets = 1  ## WT, TC, ET

val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "images_T1", "images_T2", "images_flair"]),
        transforms.AddChanneld(keys=["image", "images_T1", "images_T2", "images_flair"]),
        transforms.Orientationd(keys=["image", "images_T1", "images_T2", "images_flair"], axcodes="RAS"),
        transforms.CropForegroundd(keys=["image", "images_T1", "images_T2", "images_flair"], source_key="image"),
        transforms.NormalizeIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], nonzero=True,
                                       channel_wise=True),
        transforms.ToTensord(keys=["image", "images_T1", "images_T2", "images_flair"]),
    ]
)

post_transforms = infer_post_transforms(save_logdir, val_transform)

def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64],
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [4]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embedding=embedding)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            uncer_step = 4
            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(
                    self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96),
                                                           model_kwargs={"image": image, "embeddings": embeddings}))

            sample_return = torch.zeros((1, number_targets, 96, 96, 96))

            for index in range(4):
                #
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()

                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))

                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()

            return sample_return.to(image.device)


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)

        self.model = DiffUNet()
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96], sw_batch_size=1, overlap=0.25)

    def get_input(self, batch):
        image_dwi = batch["image"]
        images_T1 = batch["images_T1"]
        images_T2 = batch["images_T2"]
        image_flair = batch["images_flair"]
        label = batch["label"]

        label = label.float()
        return image_dwi, images_T1, images_T2, image_flair, label

    def validation_step(self, batch):
        image_dwi, image_T1, image_T2, image_flair, label = self.get_input(batch)
        image = torch.concat((image_dwi, image_T1, image_T2, image_flair), 1)
        batch['pred'] = self.window_infer(image, self.model, pred_type="ddim_sample")
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        val_labels_list_pre = decollate_batch(batch['pred'])
        val_labels_convert_pre = [
            post_pred(val_label_tensor_pre) for val_label_tensor_pre in val_labels_list_pre
        ]
        pre_one = val_labels_convert_pre[0].unsqueeze(0)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        HausdorffDistance = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)
        dice_metric(y_pred=pre_one, y=label)
        HausdorffDistance(y_pred=pre_one, y=label)
        wt = dice_metric.aggregate().item()
        Hausdorff_Distance = HausdorffDistance.aggregate().item()

        print(f"wt is {wt}")
        return [wt, Hausdorff_Distance, wt]


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)

    trainer = BraTSTrainer(env_type="pytorch",
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           val_every=val_every,
                           num_gpus=1,
                           master_port=17751,
                           training_script=__file__)

    logdir = "./logs_brats/diffusion_seg_sigmoid_all_loss_embed/model/best_model_0.5386.pt"
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=val_ds)

    print(f"v_mean is {v_mean}")