import numpy as np
from dataset.brats_data_utils_multi_label import get_loader_brats
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer_old import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from monai.transforms import AsDiscrete, Activationsd, Activations, Compose
from monai.data import CacheDataset, DataLoader, decollate_batch

set_determinism(123)
import os
from ISLES2015.networks.SwinUNETR.swin_unetr import SwinUNETR
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet
from monai.losses import DiceCELoss
from networks.UXNet_3D.network_backbone_old import UXNET
from monai.networks.nets import UNETR, AttentionUnet
from networks.SwinUNETR.swin_unetr import SwinUNETR
from networks.UNet.unet_origin import UNet
from networks.SegResNet.segresnet import SegResNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from networks.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS

data_dir = "./datasets/MICCAI_2015/"
logdir = "./logs_brats/MICCAI2015_SwinUNETR_96_seg_all_loss_embed/"

model_save_path = os.path.join(logdir, "model")

env = "pytorch"  # or env = "pytorch" if you only have one gpu.

max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 0
device = "cuda:2"

number_modality = 4
number_targets = 1  ## WT, TC, ET

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                                 sw_batch_size=1,
                                                 overlap=0.25)
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
            use_checkpoint=True,
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

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=30, max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        # self.dice_loss = DiceLoss(sigmoid=True)
        self.dice_loss = DiceLoss(softmax=True)

    def training_step(self, batch):
        image_dwi, image_T1, image_T2, image_flair, label = self.get_input(batch)
        image = torch.concat((image_dwi, image_T1, image_T2, image_flair), 1)
        pred_xstart = self.model(image)
        # loss_dice = self.dice_loss(pred_xstart, label_one)
        # loss_bce = self.bce(pred_xstart, label_one)
        # # pred_xstart = torch.sigmoid(pred_xstart)
        # loss_mse = self.mse(pred_xstart, label_one)
        # loss = loss_dice + loss_bce + loss_mse
        # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
        loss = loss_function(pred_xstart, label)
        self.log("train_loss", loss, step=self.global_step)
        return loss

    def get_input(self, batch):
        image_dwi = batch["image"]
        image_T1 = batch["images_T1"]
        image_T2 = batch["images_T2"]
        image_flair = batch["images_flair"]
        label = batch["label"]

        label = label.float()
        return image_dwi, image_T1, image_T2, image_flair, label

    def validation_step(self, batch):
        image_dwi, image_T1, image_T2, image_flair, label = self.get_input(batch)
        image = torch.concat((image_dwi, image_T1, image_T2, image_flair), 1)
        output = self.window_infer(image, self.model)
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        val_labels_list_pre = decollate_batch(output)
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
        return [wt]

    def validation_end(self, mean_val_outputs):
        wt, = mean_val_outputs
        self.log("wt", wt, step=self.epoch)
        self.log("mean_dice", (wt) / 1, step=self.epoch)

        mean_dice = (wt) / 1
        if wt > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        print(f"wt is {wt}, mean_dice is {mean_dice}")


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_loader_brats(data_dir=data_dir, batch_size=batch_size, fold=0)

    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17751,
                           training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
