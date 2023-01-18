from torch import nn
import torch
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from semantic_seg_v4 import SegmentationDataSet_3legs, three_legs
import numpy as np
import matplotlib.pyplot as plt
import os


PATH = 'model_3legs/model_3legs_tile_200_nb_classes_2_lr_0.0001_epochs_10_batch_size_8_nb_mini_batch_5.pth'
PATH = 'model_3legs/model_3legs_tile_200_nb_classes_2_lr_0.0001_epochs_20_batch_size_64_nb_mini_batch_20.pth'
PATH = 'model_3legs/model_3legs_tile_200_nb_classes_2_lr_0.0001_epochs_100_batch_size_64_nb_mini_batch_50.pth'
#PATH = 'model_3legs/CrossEntropyLoss/model_3legs_tile_200_nb_classes_2_lr_0.0001_epochs_50_batch_size_32_nb_mini_batch_500.pth'
PATH = 'Model_full_training/model_3legs_tile_200_nb_classes_2_lr_0.001_batch_size_32_epochs_3/'
model_name= 'model_epochs_2.pth'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# model=torch.load(PATH)
# model.eval()
# model.to(device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = three_legs(200,2,1)
model.load_state_dict(torch.load(PATH+model_name))
model.to(device)
print('model  eval')
model.eval()

nb_build= '239' #'225' #'269'#'239'#'437'
nb_layer= '00223' #'00100' #'00003'#'00223'#'00041'   
tile_size = 200
confiance = 0.01
powder = False
layering = True
nb_tile=(928//tile_size+1)

liste_tiles=[[nb_build,nb_layer,i] for i in range(nb_tile**2)]
image_tile=SegmentationDataSet_3legs(liste_tiles)
image_tile_dataloader = data.DataLoader(dataset=image_tile,
                                      batch_size=nb_tile**2,
                                      shuffle=False)

A, B, D, label = next(iter(image_tile_dataloader))
A, B, D, label = A.to(device), B.to(device), D.to(device), label.to(device)
# print(A[0,0])
# print(A[0,1])
# to_pil_image(A[0,0]).show()
# to_pil_image(A[0,1]).show()
if layering: #inverse the two layering and melting
    B_temp = B
    B[:,0] = B_temp[:,0]
    B[:,1] = B_temp[:,0]
    A_temp = A
    A[:,0] = A_temp[:,0]
    A[:,1] = A_temp[:,0]
# print(A[0,0])
# print(A[0,1])
# to_pil_image(A[0,0]).show()
# to_pil_image(A[0,1]).show()

output=model([A,B,D])
image = torch.zeros((928,928))

k=0
if powder:
    l=1
if not powder:
    l=0

for i in range(nb_tile):
    for j in range(nb_tile):
        if i == nb_tile-1 : #overlap if the tile_size is too big           
            if j == nb_tile-1 : #overlap if the tile_size is too big
                image[-tile_size:,-tile_size:]=output.squeeze()[k][l]
            else:
                image[-tile_size:,(j)*tile_size:(j+1)*tile_size]=output.squeeze()[k][l]
        else:
            if j == nb_tile-1 : #overlap if the tile_size is too big
                image[i* tile_size:(i+1)* tile_size,-tile_size:]=output.squeeze()[k][l] 
            else:
                image[i*tile_size:(i+1)* tile_size,(j)* tile_size:(j+1)* tile_size]=output.squeeze()[k][l]
        k+=1
mask_final=(image>confiance).float()

to_pil_image(mask_final).show()
# try: 
#     #os.mkdir('D:/ARIA/Result/'+'IMAGE_'+PATH[29:-4]+'/')    
#     os.mkdir('D:/ARIA/Result/'+'IMAGE_'+PATH[12:-4]+'/')  
# except OSError as error:
#     ahahahah=1
# #to_pil_image(mask_final).save('D:/ARIA/Result/'+'IMAGE_'+PATH[29:-4]+'/'+nb_build+'_Layer'+nb_layer+'_confiance_'+str(confiance)+'.jpg')
# to_pil_image(mask_final).save('D:/ARIA/Result/'+'IMAGE_'+PATH[12:-4]+'/'+nb_build+'_Layer'+nb_layer+'_confiance_'+str(confiance)+'.jpg')

if powder:
    to_pil_image(mask_final).save(PATH+nb_build+'_'+nb_layer+f'_confiance_{100*confiance}_powder.jpg')
if not powder and not layering:
    to_pil_image(mask_final).save(PATH+nb_build+'_'+nb_layer+f'_confiance_{100*confiance}_mask.jpg')
if layering:
    to_pil_image(mask_final).save(PATH+nb_build+'_'+nb_layer+f'_confiance_{100*confiance}_layering.jpg')
