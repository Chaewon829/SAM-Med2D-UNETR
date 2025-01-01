import torch
import torch.nn as nn
import torchvision
import sys
import os
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
    


if __name__ == '__main__':
    encoder = MedSam2dEncoder()
    # print(model)
    image = np.random.rand(2 ,4, 256, 256, 3) 
    input, features = encoder(image)
    print(f"input shape: {input.shape}")
    print(f"feautres : {len(features)}, {features[0].shape}")