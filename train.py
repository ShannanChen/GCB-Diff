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
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric

set_determinism(123)
import os
# from BraTS2020.networks.unet2 import UNet
from BraTS2020.networks.SwinUNETR.swin_unetr import SwinUNETR
from monai.networks.nets import UNETR, SwinUNETR, AttentionUnet

data_dir = "./datasets/MICCAI_2015/"
logdir = "./logs_brats/diffusion_seg_sigmoid_all_loss_embed/"

model_save_path = os.path.join(logdir, "model")

env = "pytorch"  # or env = "pytorch" if you only have one gpu.

max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 0
device = "cuda:0"

number_modality = 4
number_targets = 1  ## WT, TC, ET


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, number_modality, number_targets, [64, 64, 128, 256, 512, 64])

        self.model = BasicUNetDe(3, number_modality + number_targets, number_targets, [64, 64, 128, 256, 512, 64], act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [5]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, number_targets, 96, 96, 96),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[96, 96, 96],
                                                 sw_batch_size=1,
                                                 overlap=0.25)
        self.model = DiffUNet()
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=30, max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(softmax=True)

    def training_step(self, batch):
        image_dwi, image_T1, image_T2, image_flair, label = self.get_input(batch)
        image = torch.concat((image_dwi, image_T1, image_T2, image_flair), 1)
        x_t, t, noise = self.model(x=label, pred_type="q_sample")
        pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")
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
        output = self.window_infer(image, self.model, pred_type="ddim_sample")
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
