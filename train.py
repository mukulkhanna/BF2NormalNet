import argparse
import logging
import os
import sys
from copy import copy
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from eval import eval_net
from normal_net import NormalNet
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import SynthiaDataset

dir_checkpoint = "checkpoints/"


def train_net(
    net: nn.Module,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 2,
    lr: float = 0.0001,
    save_cp: bool = True,
    img_scale: float = 0.5,
    name: Optional[str] = None,
):

    net.train()

    if args.dataset == "synthia":
        rgb_dir = "data/synthia/rgb/{}/"
        normals_dir = "data/synthia/normals/{}/"
        seg_masks_dir = "data/synthia/seg_masks/{}/"

        train_dataset = SynthiaDataset(
            rgb_dir.format("train"),
            normals_dir.format("train"),
            seg_masks_dir.format("train"),
            img_scale,
        )

        test_dataset = SynthiaDataset(
            rgb_dir.format("test"),
            normals_dir.format("test"),
            seg_masks_dir.format("test"),
            img_scale,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
    )

    writer = SummaryWriter(
        log_dir=name, comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}"
    )

    dir_checkpoint = os.path.join(writer.get_logdir(), "checkpoints")
    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_dataset)}
        Validation size: {len(test_dataset)}
        Saving Ckpts:    {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
    )

    for epoch in range(epochs):
        net.train()
        with tqdm(
            total=len(train_dataset), desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:

                rgb = batch["rgb"].to(device)
                gt_normals = batch["normal"].to(device)
                seg_masks = batch["seg_mask"].to(device=device)

                assert rgb.shape[1] == 3, (
                    f"Network has been defined with 3 input channels, "
                    f"but loaded images have {rgb.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )
                normal_pred = net(rgb, gt_normals, seg_masks)

                ang_loss = net.loss_G_ang
                l1_loss = net.loss_G_L1

                if args.discriminator:
                    g_gan_loss = net.loss_G_GAN
                    dis_loss = net.loss_D
                else:
                    g_gan_loss = torch.Tensor([0])
                    dis_loss = torch.Tensor([0])

                total_loss = (
                    g_gan_loss.cpu()
                    + dis_loss.cpu()
                    + l1_loss.cpu()
                    + ang_loss.cpu() * 0.2
                )

                writer.add_scalar("[Train] loss/total", total_loss.item(), global_step)
                writer.add_scalar("[Train] loss/l1", l1_loss.item(), global_step)
                writer.add_scalar("[Train] loss/ang", ang_loss.item(), global_step)
                writer.add_scalar(
                    "[Train] loss/g_gan_loss", g_gan_loss.item(), global_step
                )
                writer.add_scalar("[Train] loss/dis", dis_loss.item(), global_step)

                pbar.set_postfix(
                    **{
                        "[Total loss]": total_loss.item(),
                        "[L1 loss]": l1_loss.item(),
                        "[Ang loss]": ang_loss.item(),
                        "[G_GAN loss]": g_gan_loss.item(),
                        "[Dis loss]": dis_loss.item(),
                    }
                )

                pbar.update(rgb.shape[0])
                global_step += 1

        eval_net(net, test_loader, device, writer, epoch, global_step)

        if save_cp and ((epoch + 1) % 2 == 0):
            os.makedirs(dir_checkpoint, exist_ok=True)
            torch.save(net.state_dict(), dir_checkpoint + f"/CP_epoch{epoch + 1}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the NormalNet on images and target normal maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=50,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=2,
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="LearningFalse rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        default="synthia",
        help="Dataset to be used",
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )
    parser.add_argument(
        "-di",
        "--disc",
        dest="discriminator",
        action="store_true",
        default=False,
        help="Whether discriminator needs to be trained",
    )
    parser.add_argument(
        "-n", "--name", dest="name", type=str, default=None, help="Name of experiment"
    )
    parser.add_argument(
        "--resnet",
        "-res",
        dest="resnet",
        action="store_true",
        help="Use pre-trained resnet encoder.",
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    net = NormalNet(
        discriminator=args.discriminator, bilinear=True, resnet=args.resnet,
    )
    logging.info(
        f"Network:\n"
        f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling\n'
        f"\tDiscriminator: {args.discriminator}\n"
        f'\tEncoder: {"Resnet" if args.resnet else "UNet generic"}\n'
        f"\tDataset: {args.dataset}\n"
    )

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    net.to(device=device)
    # faster convolutions, but more memory
    # torch.backends.cudnn.benchmark = True

    if args.name is not None:
        args.name = os.path.join("runs", args.name)

    try:
        train_net(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            device=device,
            img_scale=args.scale,
            name=args.name,
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
