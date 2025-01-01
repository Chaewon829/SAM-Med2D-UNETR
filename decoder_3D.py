import sys
import os
import torch
import torch.nn as nn

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



def reshape_transformer_output_3d(z, embed_dim, h, w, d, patch_size):
    batch, D, num, dim = z.shape  
    num_patches = (h // patch_size) * (w // patch_size)
    if num == num_patches + 1:  # CLS 토큰이 있는 경우
        z = z[:, :, 1:, :]  # CLS 토큰 제거
    h_patches = h // patch_size 
    w_patches = w // patch_size 
    return z.permute(0, 3, 1, 2).reshape(batch, embed_dim, h_patches, w_patches, D) #num => h * w


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
    


# === 3D 통합 모델 === #
class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
        # self.batch_size = batch_size
        # self.num_tokens = num_tokens  
        # self.embed_dim = embed_dim  #768
        # self.d = 4
    def forward(self, x):
        """
        입력 x는 Decoder에서 첫 번째 입력(z0)으로 사용됩니다.
        """
        # 더미 feature map 생성
        z3 = torch.randn(2,4,16,16,768)
        z6 = torch.randn(2,4,16,16,768)
        z9 = torch.randn(2,4,16,16,768)
        z12 = torch.randn(2,4,16,16,768)
        
        return [z3, z6, z9, z12]
    
class UNETR(nn.Module):
    def __init__(self, encoder, embed_dim=768, patch_size = 14, input_dim=3, output_dim=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = SAMDecoder3D(embed_dim,patch_size, input_dim, output_dim)

    def forward(self, x):
        features = self.encoder(x)
        print(f'encoder_features : {len(features)}, {features[0].shape}')
        x = x.permute(0,2,3,4,1) 
        output = self.decoder(x, features)
        return output
   
#Text code
if __name__ == '__main__':
    model = UNETR(encoder=DummyEncoder())
    x = torch.randn(2, 4, 3, 256, 256) # B, D, C, H, W
    print(f"inpt shape : {x.shape}")
    features = model(x)
    print(f"output shape : {features.shape}")

