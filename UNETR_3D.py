
from decoder_3D import SAMDecoder3D
from MedSAM2D_enocder import MedSam2dEncoder
import sys
import os
import torch
import torch.nn as nn


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