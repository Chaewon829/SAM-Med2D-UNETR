
import sys
import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations as A
from SAM_Med2D.segment_anything import sam_model_registry
from SAM_Med2D.segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace
from albumentations.pytorch import ToTensorV2


args = Namespace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "pretrain_model/sam-med2d_b.pth"


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(
            in_planes, 
            out_planes, 
            kernel_size=(2, 2, 1),  # (D, H, W)
            stride=(2, 2, 1),       # Stride matches kernel size
            padding=(0, 0, 0)       # No padding
        )

    def forward(self, x):
        return self.block(x)
    
class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2
        )

    def forward(self, x):
        return self.block(x)

class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class SamMed2DEncoder(nn.Module):
    def __init__(self, model):
        super(SamMed2DEncoder, self).__init__()
        self.model = model
        self.encoder = model.image_encoder
        self.devices = self.model.device
        self.reset_image()

    def forward(self, x):
        # x is expected to be [B, D, H, W, C]
        input_image, layer_features = self.set_image(x)
        return input_image, layer_features
    
    def set_image(self, image: np.ndarray, image_format: str = "RGB"):
        """
        Sets the image for the encoder.

        Parameters:
        - image (np.ndarray): Input image with shape [B, D, H, W, C].
        - image_format (str): Format of the input image ("RGB" or "BGR").

        Returns:
        - input_image (torch.Tensor): Normalized and transformed images.
        - layer_features (List[torch.Tensor]): Features from specified encoder layers with shape [B, D, ...].
        """
        assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Ensure image is float32
        image = image.astype(np.float32)

        # Normalize the image
        if self.model.pixel_mean.device.type == 'cuda':
            pixel_mean = self.model.pixel_mean.squeeze().cpu().numpy().astype(np.float32)
            pixel_std = self.model.pixel_std.squeeze().cpu().numpy().astype(np.float32)
        else:
            pixel_mean = self.model.pixel_mean.squeeze().numpy().astype(np.float32)
            pixel_std = self.model.pixel_std.squeeze().numpy().astype(np.float32)
        input_image = (image - pixel_mean) / pixel_std

        B, D, ori_h, ori_w, C = input_image.shape  # [B, D, H, W, C]
        self.original_size = (ori_h, ori_w)
        self.new_size = (self.model.image_encoder.img_size, self.model.image_encoder.img_size)
        transforms = self.transforms(self.new_size)

        # Initialize list to collect transformed images
        transformed_images = []

        for b in range(B):
            for d in range(D):
                img = input_image[b, d]  # [H, W, C]
                augmented = transforms(image=img)
                transformed_img = augmented['image']  # [C, H, W] float32
                transformed_images.append(transformed_img)

        # Stack back to [B*D, C, H, W] and ensure dtype is float32
        transformed_images = torch.stack(transformed_images).float()  # [B*D, C, H, W]

        # Move to the appropriate device
        transformed_images = transformed_images.to(self.devices)

        # Pass through the encoder
        x = self.encoder.patch_embed.proj(transformed_images)  # [B*D, embed_dim, H', W']
        x = x.permute(0, 2, 3, 1)  # [B*D, H', W', embed_dim]
        
        layer_features = []
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i + 1 in [3, 6, 9, 12]:
                # print(f"Block {i+1} output shape: {x.shape}")  # Debugging
                layer_features.append(x)
        
        # Reshape layer_features from [B*D, ...] to [B, D, ...]
        reshaped_features = []
        for feat in layer_features:
            # feat shape: [B*D, H', W', C']
            new_shape = (B, D) + feat.shape[1:]
            reshaped_feat = feat.view(B, D, *feat.shape[1:])  # [B, D, H', W', C']
            reshaped_features.append(reshaped_feat)
            # print(f"Reshaped feature shape: {reshaped_feat.shape}")  # Debugging
    
        self.features = reshaped_features
        self.is_image_set = True
        return  self.features

    def transforms(self, new_size):
        return A.Compose([
            A.Resize(int(new_size[0]), int(new_size[1]), interpolation=cv2.INTER_NEAREST),
            ToTensorV2(p=1.0)
        ], p=1.0)
    
    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.new_size = None


class MedSam2dEncoder(nn.Module):
    def __init__(self):
        super(MedSam2dEncoder, self).__init__()
        self.model = sam_model_registry["vit_b"](args).to(device)
        self.sam = SamMed2DEncoder(self.model)       
        
    def forward(self, x):   
        feature = self.sam.set_image(x)        
        return x, feature


class SAMDecoder3D(nn.Module):
    def __init__(self, embed_dim=768, patch_size=16, input_dim=3, output_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Decoder layers
        self.decoder0 = nn.Sequential(
            Conv3DBlock(input_dim, 32, 3),
            Conv3DBlock(32, 64, 3)
        )

        self.decoder3 = nn.Sequential(
            Deconv3DBlock(embed_dim, 512),
            Deconv3DBlock(512, 256),
            Deconv3DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv3DBlock(embed_dim, 512),
            Deconv3DBlock(512, 256)
        )

        self.decoder9 = Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv3DBlock(1024, 512),
            Conv3DBlock(512, 512),
            SingleDeconv3DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv3DBlock(512, 256),
            Conv3DBlock(256, 256),
            SingleDeconv3DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv3DBlock(256, 128),
            Conv3DBlock(128, 128),
            SingleDeconv3DBlock(128, 64),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.decoder0_header = nn.Sequential(
            Conv3DBlock(128, 64),
            Conv3DBlock(64, 64),
            SingleConv3DBlock(64, output_dim, 1)
        )

    def forward(self, x, features):
        z0, z3, z6, z9, z12 = x, *features
        B, D, C, H, W = x.shape    
        # print(f"z0 : {z0.shape}, z3 : {z3.shape}, z6 : {z6.shape}, z9 : {z9.shape}, z12 : {z12.shape}")
        # z0 = z0.permute(0,2,3,4,1) #B, C, H, W, D
        z3 = z3.permute(0,4,2,3,1) #B, 768, h, w, D
        z6 = z6.permute(0,4,2,3,1) #B, 768, h, w, D
        z9 = z9.permute(0,4,2,3,1) #B, 768, h, w, D
        z12 = z12.permute(0,4,2,3,1) #B, 768, h, w, D
        # print(f"z0 : {z0.shape}, z3 : {z3.shape}, z6 : {z6.shape}, z9 : {z9.shape}, z12 : {z12.shape}")

        # Decoder operations
        z12 = self.decoder12_upsampler(z12)  # 512
        z9 = self.decoder9(z9)  # 512
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1)) 
        z6 = self.decoder6(z6)  # 256
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1)) 
        z3 = self.decoder3(z3)  # 128
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1)) 
        z0 = self.decoder0(z0)  # 64
        output = self.decoder0_header(torch.cat([z0, z3], dim=1)) 
        return output


class M3dSAM2DUNETR(nn.Module) :
    def __init__(self, embed_dim = 768, patch_size = 16, input_dim = 3, output_dim=3):
        super(M3dSAM2DUNETR, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = MedSam2dEncoder() 
        self.decoder = SAMDecoder3D(embed_dim=self.embed_dim, patch_size=self.patch_size, input_dim=self.input_dim, output_dim=self.output_dim)

        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):

        B, D, C, H, W = x.shape
        x = x.view(B * D, C, H, W).contiguous()
        features = self.encoder(x)        
        print(f'encoder_features : {len(features)}, {features[0].shape}')       
        reshaped_features = [
            f.view(B, D, f.shape[1], f.shape[2], f.shape[3]).contiguous() for f in features
        ]
        output = self.decoder(x.view(B, D, C, H, W).contiguous(), reshaped_features)
        return output


#test

if __name__ == '__main__':
    model = M3dSAM2DUNETR()
    # x = torch.randn( 1, 4, 3, 1024,1024 ) #B, D, C, H, W
    x = torch.randn( 1, 4, 3, 224,224 ) #B, D, C, H, W
    output = model(x)
    
    print(f"UNETR output_shape : {output.shape}")