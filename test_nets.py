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
from ISLES2015.networks.unet2 import UNet
from ISLES2015.networks.SwinUNETR.swin_unetr import SwinUNETR
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet
from load_datasets_transforms import test_data_loader, data_transforms, infer_post_transforms
from monai import transforms, data
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Activationsd, Activations, Compose
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceCELoss
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
from networks.UXNet_3D.network_backbone_old import UXNET

set_determinism(123)
import os

data_dir = "./datasets/MICCAI_2015/"
save_logdir = "./logs_brats/MICCAI2015_SwinUNETR_96_seg_all_loss_embed/nii"

max_epoch = 300
batch_size = 1
val_every = 10
device = "cuda:2"

number_modality = 4
number_targets = 1  ## WT, TC, ET

val_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "images_T1", "images_T2", "images_flair"]),
        transforms.AddChanneld(keys=["image", "images_T1", "images_T2", "images_flair"]),
        transforms.Orientationd(keys=["image", "images_T1", "images_T2", "images_flair"], axcodes="RAS"),
        transforms.CropForegroundd(keys=["image", "images_T1", "images_T2", "images_flair"], source_key="image"),
        transforms.NormalizeIntensityd(keys=["image", "images_T1", "images_T2", "images_flair"], nonzero=True, channel_wise=True),
        transforms.ToTensord(keys=["image", "images_T1", "images_T2", "images_flair"]),
    ]
)

post_transforms = infer_post_transforms(save_logdir, val_transform)


def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)

        # self.model = UNETR(
        #     in_channels=number_modality,
        #     out_channels=number_targets,
        #     img_size=(96, 96, 96),
        #     feature_size=16,
        #     hidden_size=768,
        #     mlp_dim=3072,
        #     num_heads=12,
        #     pos_embed="perceptron",
        #     norm_name="instance",
        #     res_block=True,
        #     dropout_rate=0.0,
        # ).to(device)
        # self.model = UNet(n_channels=number_modality, n_classes=number_targets, width_multiplier=1, trilinear=True, use_ds_conv=False).to(device)
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=number_modality,
            out_channels=number_targets,
            feature_size=96,
            use_checkpoint=False,
        ).to(device)
        # _, self.model = TransBTS(dataset='feta', _conv_repr=True, _pe_type='learned')
        # self.model = self.model.to(device)
        # self.model = UXNET(
        #     in_chans=number_modality,
        #     out_chans=number_targets,
        #     depths=[2, 2, 2, 2],
        #     feat_size=[48, 96, 192, 384],
        #     drop_path_rate=0,
        #     layer_scale_init_value=1e-6,
        #     spatial_dims=3,
        # ).to(device)

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
        image_dwi, images_T1, images_T2, image_flair, label = self.get_input(batch)
        image = torch.concat((image_dwi, images_T1, images_T2, image_flair), 1)
        batch['pred'] = self.window_infer(image,  self.model)
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        val_labels_list_pre = decollate_batch(batch['pred'])
        val_labels_convert_pre = [
            post_pred(val_label_tensor_pre) for val_label_tensor_pre in val_labels_list_pre
        ]
        pre_one = val_labels_convert_pre[0].unsqueeze(0)
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        HausdorffDistance = HausdorffDistanceMetric(include_background=True, reduction='mean', percentile=95)
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

    logdir = "./logs_brats/MICCAI2015_SwinUNETR_96_seg_all_loss_embed/model/best_model_0.6010.pt"
    trainer.load_state_dict(logdir)
    v_mean, _ = trainer.validation_single_gpu(val_dataset=val_ds)

    print(f"v_mean is {v_mean}")