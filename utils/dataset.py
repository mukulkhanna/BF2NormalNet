import logging
import os
from glob import glob
from os import listdir
from os.path import splitext

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class SynthiaDataset(Dataset):
    def __init__(
        self, rgb_dir: str, normals_dir: str, seg_masks_dir: str, scale: float = 0.5,
    ):
        self.rgb_dir = rgb_dir
        self.normals_dir = normals_dir
        self.seg_masks_dir = seg_masks_dir
        self.scale = scale

        assert 0 < scale <= 1, "Scale must be between 0 and 1"

        self.ids = [
            splitext(file)[0]
            for file in sorted(listdir(rgb_dir))
            if not file.startswith(".")
        ]

        logging.info(f"Creating {rgb_dir} dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, "Scale is too small"
        pil_img = pil_img.resize((newW, newH), Image.NEAREST)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def preprocess_normals(self, normal_map):
        h, w = normal_map.shape[0:2]
        normal_map = cv2.resize(
            normal_map, dsize=(int(w * self.scale), int(h * self.scale))
        )
        normal_map = normal_map.transpose((2, 0, 1))
        normal_map = normal_map / 2 + 0.5
        return normal_map

    def __getitem__(self, i):

        idx = self.ids[i]
        rgb_file = sorted(glob(self.rgb_dir + idx + ".*"))
        normal_file = sorted(glob(self.normals_dir + idx + ".*"))
        seg_mask_file = glob(self.seg_masks_dir + idx + ".*")

        assert (
            len(normal_file) == 1
        ), f"Either no normal map or multiple maps found for the ID {idx}: {normal_file}: {rgb_file}"
        assert (
            len(rgb_file) == 1
        ), f"Either no image or multiple images found for the ID {idx}: {rgb_file}"

        rgb = Image.open(rgb_file[0])
        normal_map = np.load(normal_file[0])[:, :, ::-1]
        seg_mask = Image.open(seg_mask_file[0]).convert("L")

        assert (
            rgb.size == normal_map.shape[0:2][::-1] == seg_mask.size
        ), f"Image, normal map, and segmentation mask {idx} should be the same size, but are {img.size} and {mask.shape[0:2]}"

        rgb = self.preprocess(rgb, self.scale)
        normal_map = self.preprocess_normals(normal_map)
        seg_mask = self.preprocess(seg_mask, self.scale)

        return {
            "rgb": torch.from_numpy(rgb).float(),
            "normal": torch.from_numpy(normal_map).float(),
            "seg_mask": torch.from_numpy(seg_mask).type(torch.uint8),
        }
