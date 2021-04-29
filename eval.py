import logging
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from normal_net import NormalNet
from torch.utils.tensorboard import SummaryWriter


def eval_net(
    net: NormalNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    writer: Optional[SummaryWriter],
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    ckpt: Optional[str] = None,
    save: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    net.eval()
    n_val = len(loader)  # number of batches

    # accuracy stats
    l1_loss = 0.0
    mean_ang_loss = 0.0
    median_ang = 0.0
    ctr, avg_p11, avg_p22, avg_p30, avg_p60 = 0, 0.0, 0.0, 0.0, 0.0
    bg_only_imgs = 0

    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for batch in loader:
            rgb, gt_normals, seg_masks = (
                batch["rgb"].to(device),
                batch["normal"].to(device),
                batch["seg_mask"].to(device),
            )

            with torch.no_grad():
                pred_normals = net.test(rgb)
                l1_loss += net.l1_criterion(pred_normals, gt_normals).item()
                ang_loss, median, p11, p22, p30, p60, bg_only = net.calculate_ang_loss(
                    pred_normals, gt_normals, seg_masks, mode="eval"
                )

                if bg_only:
                    bg_only_imgs += 1
                else:
                    median_ang += median.item()
                    mean_ang_loss += ang_loss.item()

                    if ckpt is not None and save:
                        for idx, i in enumerate(pred_normals):
                            pred = i.cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
                            img = rgb[idx].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
                            cv2.imwrite(f"results/{ckpt}/{ctr}_pred.png", pred * 255)
                            cv2.imwrite(f"results/{ckpt}/{ctr}_rgb.png", img * 255)
                            true = (
                                gt_normals[idx]
                                .cpu()
                                .numpy()
                                .transpose(1, 2, 0)[:, :, ::-1]
                            )
                            cv2.imwrite(f"results/{ckpt}/{ctr}_gt.png", true * 255)
                            # np.savez(f'results/{ckpt}/{ctr}_raw_normals', true) # save normal map np matrix
                            ctr += 1

                    avg_p11 += p11
                    avg_p22 += p22
                    avg_p30 += p30
                    avg_p60 += p60

            pbar.update()

        n_val = n_val - bg_only_imgs

        l1_test_loss = l1_loss / n_val
        ang_test_loss = mean_ang_loss / n_val
        median_ang = median_ang / n_val
        total_test_loss = l1_test_loss + ang_test_loss * 0.2

        logging.info("[Test] Total loss: {:.4f}".format(total_test_loss))
        logging.info("[Test] L1 loss: {:.4f}".format(l1_test_loss))
        logging.info(
            "[Test] Mean angular distance: {:.4f} rad ({:.2f}°)".format(
                ang_test_loss, ang_test_loss * 180 / np.pi
            )
        )
        logging.info(
            "[Test] Median angular distance: {:.4f} rad ({:.2f}°)".format(
                median_ang, median_ang * 180 / np.pi
            )
        )
        logging.info("% of angles less than 11.25 deg: {:.2f}%".format(p11))
        logging.info("% of angles less than 22.5 deg: {:.2f}%".format(p22))
        logging.info("% of angles less than 30 deg: {:.2f}%".format(p30))
        logging.info("% of angles less than 60 deg: {:.2f}%".format(p60))

        if writer is not None:
            if epoch == 0:
                writer.add_images("images", rgb, epoch)
                writer.add_images("normal maps/true", gt_normals, epoch)

            writer.add_images("normal maps/pred", pred_normals, epoch)
            writer.add_scalar("[Test] loss/l1", l1_test_loss, global_step)
            writer.add_scalar("[Test] loss/ang", ang_test_loss, global_step)
            writer.add_scalar("[Test] loss/total", total_test_loss, global_step)

    net.train()
