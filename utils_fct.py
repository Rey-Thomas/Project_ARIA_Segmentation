import torch 
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import os
from PIL import Image
import matplotlib as plt

class DownConvulution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownConvulution,self).__init__()
        self.conv=nn.Conv2d(in_channels, out_channels, kernel_size = 5, stride=1, padding = 2, padding_mode = 'reflect')
        self.relu=nn.ReLU()
        self.BN=nn.BatchNorm2d(out_channels)
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.BN(x)


        return x

class UpConvulution(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpConvulution,self).__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels[0], kernel_size = 3, stride=1)
        self.relu=nn.ReLU()
        self.upsampleNN= nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2=nn.Conv2d(out_channels[0], out_channels[1], kernel_size = 1, stride=1, padding = 2, padding_mode = 'reflect')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.upsampleNN(x)
        x = self.conv2(x)
        x = self.relu(x)

        return x

def supperposition_mask_on_image(output, data_image = ['228','00235','5'], confiance=0.6,path_data= 'D:/ARIA/Dataset/Data_3_legs/', tile_size = 200, which_image= 'melting'):
    directory_build_layer = data_image[0]+'_Layer'+data_image[1]

    path_tile=path_data+directory_build_layer+'/'+f'tile_size={tile_size}/{data_image[2]}/'
    for elt in os.listdir(path_tile):
            if 'layering' in elt and  which_image == 'layering':
                crop_image = Image.open(path_tile+elt)
            if 'melting' in elt and  which_image == 'melting':
                crop_image = Image.open(path_tile+elt)

    mask_final=(output>confiance).float()
    mask_piece = to_pil_image(mask_final.squeeze()[0])
    mask_poudre = to_pil_image(mask_final.squeeze()[1])

    mask = Image.new("L", crop_image.size, 192)
    im = Image.composite(crop_image, mask_poudre, mask)
    im = Image.composite(im,mask_piece, mask)
    #im = Image.composite(crop_image, mask_piece,mask)
    return im


def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def image_creation(im1, im2, nb_tile):
    if nb_tile in ['5','10','15','20']:
        image = get_concat_v_blank(im1, im2)
    else:
        image = get_concat_h_blank(im1, im2)
    return image


