""" Full assembly of the parts to form the complete network """

import functools
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .unet_parts import *


def nanmean(
    v: torch.Tensor, *args: Any, inplace: bool = False, **kwargs: Any
) -> torch.Tensor:
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class NormalNet(nn.Module):
    def __init__(
        self, discriminator: bool = False, bilinear: bool = True, resnet: bool = False,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        self.discriminator_exists = discriminator

        self.generator = Generator(bilinear=bilinear, resnet=resnet,)

        if self.discriminator_exists:
            self.discriminator = NLayerDiscriminator(input_nc=6)
            self.optimizer_D = torch.optim.Adam(
                self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
            )
            self.criterionGAN = GANLoss()

        # self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=0.001)

        self.l1_criterion = nn.L1Loss()

    def calculate_ang_percents(self, acos: np.ndarray, seg_masks: np.ndarray):
        gt_foreground_pixels = np.count_nonzero(~np.isnan(seg_masks))

        if gt_foreground_pixels == 0:  # no buildings present
            bg_only = True
            p11, p22, p30, p60 = 0, 0, 0, 0
        else:
            c11 = np.count_nonzero((acos * 180 / np.pi) < 11.25)
            c22 = np.count_nonzero((acos * 180 / np.pi) < 22.5)
            c30 = np.count_nonzero((acos * 180 / np.pi) < 30)
            c60 = np.count_nonzero((acos * 180 / np.pi) < 60)

            p11 = c11 * 100 / gt_foreground_pixels
            p22 = c22 * 100 / gt_foreground_pixels
            p30 = c30 * 100 / gt_foreground_pixels
            p60 = c60 * 100 / gt_foreground_pixels

            bg_only = False

        return p11, p22, p30, p60, bg_only

    def calculate_ang_loss(
        self,
        pred_normals: torch.Tensor,
        gt_normals: torch.Tensor,
        seg_masks: torch.Tensor,
        mode: Optional[str] = None,
    ):

        # mapping to [-1, +1]
        gt_normals = gt_normals * 2 - 1
        pred_normals = pred_normals * 2 - 1

        nz_pred_normals = (
            pred_normals / torch.norm(pred_normals, dim=1)[:, np.newaxis, :, :]
        )

        dot_product = torch.sum(gt_normals * nz_pred_normals, dim=1)
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        acos = torch.acos(dot_product)

        if mode == "eval":
            seg_masks = seg_masks.float()
            seg_masks[seg_masks == 0] = float(
                "nan"
            )  # excluding predictions for background (non-building) regions
            acos = acos[:, np.newaxis, :, :] * seg_masks
            mean_ang_loss = nanmean(acos)
            acos = acos.detach().cpu().numpy()
            median_ang = np.nanmedian(acos)

            p11, p22, p30, p60, bg_only = self.calculate_ang_percents(
                acos, seg_masks.detach().cpu().numpy()
            )

            return mean_ang_loss, median_ang, p11, p22, p30, p60, bg_only
        else:
            seg_masks[
                seg_masks == 0
            ] = 1e-5  # reduced weights for background (non-building) regions
            acos = acos[:, np.newaxis, :, :] * seg_masks
            mean_ang_loss = torch.mean(acos)
            return mean_ang_loss

    def set_requires_grad(
        self,
        nets: Union[List[nn.Module], nn.Module],
        requires_grad: Optional[bool] = False,
    ) -> None:
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(
        self, rgb: torch.Tensor, gt_normals: torch.Tensor, seg_masks: torch.Tensor
    ) -> torch.Tensor:
        pred_normals = self.generator(rgb)

        if not self.discriminator_exists:
            self.loss_G_L1 = self.l1_criterion(pred_normals, gt_normals)
            self.loss_G_ang = self.calculate_ang_loss(
                pred_normals, gt_normals, seg_masks
            )

            self.loss_G = self.loss_G_L1 + self.loss_G_ang * 0.2
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()
            return pred_normals

        # Fake; stop backprop to the generator by detaching fake_B
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_D.zero_grad()
        fake_AB = torch.cat((pred_normals, rgb), 1)
        pred_fake = self.discriminator(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((rgb, gt_normals), 1)
        pred_real = self.discriminator(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(
            self.discriminator, False
        )  # D requires no gradients when optimizing G

        self.optimizer_G.zero_grad()  # set G's gradients to zero
        # calculate graidents for G
        fake_AB = torch.cat((rgb, pred_normals), 1)
        pred_fake = self.discriminator(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.l1_criterion(pred_normals, gt_normals)
        self.loss_G_ang = (
            self.calculate_ang_loss(pred_normals, gt_normals, seg_masks) * 0.2
        )
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_ang
        self.loss_G.backward()
        # udpate G's weights
        self.optimizer_G.step()

        return pred_normals

    def test(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.generator(rgb)


class Generator(nn.Module):
    def __init__(self, bilinear: bool = True, resnet: bool = False,) -> None:
        super().__init__()

        factor = 2 if bilinear else 1
        self.resnet = resnet

        if resnet:
            self.resnet_encoder = torch.hub.load(
                "pytorch/vision:v0.6.0", "resnet18", pretrained=True
            )
            self.resnet_encoder.cuda().eval()
            # additional upsampling layers reqd with resnet encoder
            self.up5 = Up(128, 64, bilinear)
            self.up6 = Up(128, 64, bilinear)
        else:  # UNet encoder
            self.inc = DoubleConv(3, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.resnet:
            x0 = self.resnet_encoder.conv1(x)
            x1 = self.resnet_encoder.bn1(x0)
            x1 = self.resnet_encoder.relu(x1)
            # try passing x1 above to self.up5
            x1 = self.resnet_encoder.maxpool(x1)
            x2 = self.resnet_encoder.layer1(x1)
            x3 = self.resnet_encoder.layer2(x2)
            x4 = self.resnet_encoder.layer3(x3)
            x5 = self.resnet_encoder.layer4[0](x4)
            x6 = self.resnet_encoder.layer4[1](x5)

            up1 = self.up1(x6, x5)
            up2 = self.up2(up1, x4)
            up3 = self.up3(up2, x3)
            up4 = self.up4(up3, x2)
            up5 = self.up5(up4, x0)
            x = self.up6.up(up5)

        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        x = self.outc(x)
        x = torch.clamp(x, 0, 1)
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator. Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""

    def __init__(self, input_nc, ndf: int = 64, n_layers: int = 3):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super().__init__()

        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=True,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=True,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLoss(nn.Module):
    """Define different GAN objectives. Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(
        self,
        gan_mode: str = "lsgan",
        target_real_label: float = 1.0,
        target_fake_label: float = 0.0,
    ) -> None:
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction: torch.Tensor, target_is_real: bool):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
