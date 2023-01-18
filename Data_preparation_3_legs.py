from torch import nn
import torch
from torch.utils import data
from PIL import Image
from torchvision import datasets, transforms
import os
import numpy as np
import pandas as pd

def tile_creation(image:torch.tensor, image_size:int, tile_size:int):
    B=[]
    for i in range(image_size//tile_size+1):
        for j in range(image_size//tile_size+1): 
            if i == image_size//tile_size : #overlap if the tile_size is too big           
                if j == image_size//tile_size : #overlap if the tile_size is too big
                    B.append(image[-tile_size:,-tile_size:])
                else:
                    B.append(image[-tile_size:,(j)*tile_size:(j+1)*tile_size])    
            else:
                if j == image_size// tile_size : #overlap if the tile_size is too big
                    B.append(image[i* tile_size:(i+1)* tile_size,-tile_size:])
                else:
                    B.append(image[i*tile_size:(i+1)* tile_size,(j)* tile_size:(j+1)* tile_size])
    return B


def Data_creation(path:str,nb_build:str, nb_layer:str, directory_data:str, tile_size:int ,directory_mask:str ):
    if nb_layer>='474':
        melting_image_path=path+nb_build+'_Layer'+nb_layer+'_Visible_MeltingEnd_001.png'
        layering_image_path=path+nb_build+'_Layer'+nb_layer+'_Visible_LayeringEnd_001.png'
        mask_path=directory_mask[0]+nb_build+'_Layer'+nb_layer+'_Visible_MeltingEnd.png'
        mask_inverse_path=directory_mask[1]+nb_build+'_Layer'+nb_layer+'_Visible_MeltingEnd.png'
    if nb_layer<'474':
        melting_image_path=path+nb_build+'_Layer'+nb_layer+'_Visible_MeltingEnd.jpg'
        layering_image_path=path+nb_build+'_Layer'+nb_layer+'_Visible_LayeringEnd.jpg'
        mask_path=directory_mask[0]+nb_build+'_Layer'+nb_layer+'_Visible_MeltingEnd.jpg'
        mask_inverse_path=directory_mask[1]+nb_build+'_Layer'+nb_layer+'_Visible_MeltingEnd.jpg'


    try : 
        melting_image = Image.open(melting_image_path)
    except OSError as error:
        return print(f'Could not find the image in {melting_image_path}')
    try : 
        layering_image = Image.open(layering_image_path)
    except OSError as error:
        return print(f'Could not find the image in {layering_image_path}')
    try: 
        os.mkdir(directory_data+nb_build+'_Layer'+nb_layer+'/')    
    except OSError as error:
        print("Directory layer can not be created")
    # melting_image = Image.open(melting_image_path)
    # layering_image = Image.open(layering_image_path)
    mask = Image.open(mask_path)
    mask_inverse = Image.open(mask_inverse_path)
    trans=transforms.Compose([
            #grayscale
            transforms.Grayscale(num_output_channels=1),
            #to resize
            transforms.Resize(size=(928,928)),
            transforms.ToTensor()
            ])
    trans_pil_image= transforms.ToPILImage()
    melting_image=trans(melting_image)
    layering_image=trans(layering_image)
    mask=trans(mask)
    mask_inverse = trans(mask_inverse)

    
    if nb_layer<'474':
        trans_pil_image(melting_image).save(directory_data+nb_build+'_Layer'+nb_layer+'/'+melting_image_path[-37:])
        trans_pil_image(layering_image).save(directory_data+nb_build+'_Layer'+nb_layer+'/'+layering_image_path[-38:])
    if nb_layer>='474':
        trans_pil_image(melting_image).save(directory_data+nb_build+'_Layer'+nb_layer+'/'+melting_image_path[-41:-8]+melting_image_path[-4:])
        trans_pil_image(layering_image).save(directory_data+nb_build+'_Layer'+nb_layer+'/'+layering_image_path[-42:-8]+layering_image_path[-4:])
    melting_image_path[-41:-8]+melting_image_path[-4:]
    trans_pil_image(mask).save(directory_data+nb_build+'_Layer'+nb_layer+'/'+nb_build+'_Layer'+nb_layer+'_Visible_Mask.jpg')
    trans_pil_image(mask_inverse).save(directory_data+nb_build+'_Layer'+nb_layer+'/'+nb_build+'_Layer'+nb_layer+'_Visible_Mask_inverse.jpg')

    try: 
        os.mkdir(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/')    
    except OSError as error:
        ahahahah=1
        #print("Directory tile size can not be created")
    
    crop_melting_image=tile_creation(melting_image.squeeze(),928,tile_size)
    crop_layering_image=tile_creation(layering_image.squeeze(),928,tile_size)
    crop_mask=tile_creation(mask.squeeze(),928,tile_size)
    crop_mask_inverse=tile_creation(mask_inverse.squeeze(),928,tile_size)

    D=[]
    #Création de la matrice des coordonnées normalisée
    coord_x=torch.tensor(np.linspace(-928/2,928/2,928)/(928/2))
    coord_x=coord_x*torch.ones((928,928))

    coord_y = torch.reshape(torch.tensor(-np.linspace(-928/2,928/2,928)/(928/2)),(928,1))
    coord_y=coord_y*torch.ones((928,928))   

    D_x=tile_creation(coord_x,928,tile_size)
    D_y=tile_creation(coord_y,928,tile_size)


    #for i in range(len(crop_melting_image)):
    for i in range(len(crop_mask_inverse)): #à enlever
        try: 
            os.mkdir(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/')    
        except OSError as error:
            ahahahha=1
            #print(f"Directory {i} can not be created")


        trans_pil_image(crop_melting_image[i]).save(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/crop_melting.jpg')
        trans_pil_image(crop_layering_image[i]).save(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/crop_layering.jpg')
        trans_pil_image(crop_mask[i]).save(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/crop_mask.jpg')
        trans_pil_image(crop_mask_inverse[i]).save(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/crop_mask_inverse.jpg')
        

        temp_np = D_x[i].numpy() #convert to Numpy array
        df = pd.DataFrame(temp_np) #convert to a dataframe
        df.to_csv(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/localisation_x.csv',index=False) #save to file

        temp_np = D_y[i].numpy() #convert to Numpy array
        df = pd.DataFrame(temp_np) #convert to a dataframe
        df.to_csv(directory_data+nb_build+'_Layer'+nb_layer+f'/tile_size={tile_size}/{i}/localisation_y.csv',index=False) #save to file



    return #print('Dataset Done')



# for i in range(318):
#     if i>16:#on prends pas les premières couches
#         if i<10:
#             layer=f'0000{i}'
#         if i<100:
#             layer=f'000{i}'
#         if 99<i<1000:
#             layer=f'00{i}'
#         if i>=1000:
#             layer=f'0{i}'
#         if i%10 ==0: 
#             print(i)
#         Data_creation(path='D:/ARIA/Dataset/Image/Photo_recadree/',nb_build='507', nb_layer=layer, directory_data='D:/ARIA/Dataset/Data_3_legs/', tile_size=200, directory_mask=['D:/ARIA/Dataset/Mask_parfait/','D:/ARIA/Dataset/Mask_inverse/'])

Data_creation(path='D:/ARIA/Dataset/Image/Photo_recadree/',nb_build='269', nb_layer='00003', directory_data='D:/ARIA/Dataset/Data_3_legs/', tile_size=200, directory_mask=['D:/ARIA/Dataset/Mask_parfait/','D:/ARIA/Dataset/Mask_inverse/'])

