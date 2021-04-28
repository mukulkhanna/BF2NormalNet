import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from eval import eval_net
from normal_net import NormalNet
from utils.dataset import HolicityDataset, SynthiaDataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict normal maps for test images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "-m",
        default="MODEL.pth",
        metavar="FILE",
        help="Specify the file in which the model is stored",
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
        "--save",
        "-sv",
        dest="save",
        action="store_true",
        help="Save the results",
        default=False,
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        help="Scale factor for the input images",
        default=0.5,
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
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.dataset == "synthia":
        rgb_dir = "data/synthia/rgb/{}/"
        normals_dir = "data/synthia/normals/{}/"
        seg_masks_dir = "data/synthia/seg_masks/{}/"

        test_dataset = SynthiaDataset(
            rgb_dir.format("test"),
            normals_dir.format("test"),
            seg_masks_dir.format("test"),
            args.scale,
        )

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
    )

    ckpt = args.model[5:]
    if args.save:
        os.makedirs(f"results/{ckpt}", exist_ok=True)  # for saving results

    net = NormalNet(discriminator=False, bilinear=True, resnet=args.resnet,)

    logging.info("Loading model {}".format(args.model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    net.to(device=device)

    model_dict = net.state_dict()
    pretrained_dict = torch.load(args.model, map_location=device)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.load_state_dict(pretrained_dict)

    l1_criterion = nn.L1Loss()
    logging.info("Model loaded. Starting evaluation.")

    eval_net(
        net, test_loader, device, writer=None, ckpt=ckpt, save=args.save,
    )
