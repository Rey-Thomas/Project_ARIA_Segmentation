# import os
# import torch
# import matplotlib.pyplot as plt
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp

# from pprint import pprint
# from torch.utils.data import DataLoader


# import torchvision.transforms.functional as TF
# from torchvision.transforms.functional import to_pil_image

# import numpy as np
# import random
# import os
# from glob import glob
# import cv2

# class YourCustomDataset(Dataset):
#     """ 
#     Allows to read the input and target data, 
#     combine them to a pair of tensors which then can be passed to specific dataloaders 
#     and finally to the train / validate or test the model 
#     Args:
#         Dataset (object)
        
#     """
#     def __init__(self, root_path, ipt, tgt, train_transform=None):
#         """
#         NEEDED: Constructor for dataset class for semantic segmentation.
#         https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
#         Args:
#             root_path (str): path to data
#             ipt (str): specific path to input data 
#             tgt (str): specific path to target (mask) data
#             train_transform (bool, optional): Applies the defined data transformations used in training. Defaults to None.
#         """
#         super(YourCustomDataset, self).__init__()
#         self.root_path = root_path
#         self.ipt = ipt
#         self.tgt = tgt
#         self.train_transform = train_transform
        
#     def my_train_segmentation_transforms(self, input_patch, tgt_patch): 
#         """
#         Function that applies specific, reproducible transformations to single patches of the input and target data for training. 
#         IMPORTANT FEATURE: 
#         The transformations are applied to the input and target patch in the same way. 
        
#         Args:
#             input_patch (np.array): input data array (image) 224x224 pixels
#             tgt_patch (np.array): target data array (image) 224x224 pixels
#         Returns:
#             set of float 32 tensors: random combinations of the transformations listed below
#         """
#         input_patch = TF.to_tensor(input_patch)
#         tgt_patch = TF.to_tensor(tgt_patch)
        
#         if random.random() > 0.5:
#             # create random angle between -60 and 60 degrees
#             angle = random.randint(-60, 60)
#             # apply to both input and target patch
#             input_patch = TF.rotate(input_patch, angle)
#             tgt_patch = TF.rotate(tgt_patch, angle)
            
#         if random.random() > 0.5: 
#             # flip the patches horizontally
#             input_patch = TF.hflip(input_patch)
#             tgt_patch = TF.hflip(tgt_patch)
            
#         if random.random() > 0.5: 
#             # flip the patches vertically
#             input_patch = TF.vflip(input_patch)
#             tgt_patch = TF.vflip(tgt_patch)
            
#         return input_patch, tgt_patch

#     def my_test_segmentation_transforms(self, input_patch, tgt_patch):
#         """
#         In testing, there is no specific need to transform the data, rotate or flip it. 
#         Therefore, this function only creates a tensor object from the np.array input/target data .
#         Args:
#             input_patch (np.array): input data array (image) 224x224 pixels
#             tgt_patch (np.array): target data array (image) 224x224 pixels
#         Returns:
#             set of float 32 tensors: input and target patch
#         """

#         input_patch = TF.to_tensor(input_patch)
#         tgt_patch = TF.to_tensor(tgt_patch)
        
#         return input_patch, tgt_patch
        
#     def __len__(self):
#         """
#         Necessary function returning the length of the dataset
#         https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
#         Returns:
#             _type_: _description_
#         """
#         l1 = os.listdir(os.path.join(self.root_path, self.ipt)) # dir is your directory path
#         l2 = os.listdir(os.path.join(self.root_path, self.tgt)) # dir is your directory path
#         number_files_inp = len(l1)
#         number_files_tgt = len(l2)

#         if number_files_inp == number_files_tgt:
#             return number_files_inp

#     def __getitem__(self, idx):
#         """
#         Necessary function that loads and returns a sample from the dataset at a given index. 
#         https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
#         Based on the index,it identifies the input and target images location on the disk, 
#         reads both items as a numpy array (float32). If the train_transform argument is True, 
#         the above defined train transformations are applied. Else, the test transformations are applied
#         Args:
#             idx (iterable): 
#         Returns:
#             tensors: input and target image
#         """
#         # change here so that in the class the scale can be defined not hard coded 'geb_' or 'geb_10'
#         # img_path_input_patch = os.path.join(self.root_path, self.ipt, ".jpg")#f"geb_{idx}.npy")
#         # img_path_tgt_patch = os.path.join(self.root_path, self.tgt, '.jpg')#f"geb_{str(self.tgt_scale)}_{idx}.npy")
        
#         img_path_input_patch = glob(self.root_path+self.ipt+"*.jpg")
#         img_path_input_patch=img_path_input_patch[idx]
#         img_path_tgt_patch = self.root_path+self.tgt+img_path_input_patch[-37:-4]+'.jpg'
#         #tgt_patch = np.load(img_path_tgt_patch).astype('float32')
#         #input_patch = np.load(img_path_input_patch).astype('float32')
#         input_patch=cv2.imread(img_path_input_patch)
#         tgt_patch = cv2.imread(img_path_tgt_patch)

            
#         if self.train_transform:
#             input_patch, tgt_patch = self.my_train_segmentation_transforms(input_patch, tgt_patch)
        
#         else: 
#             input_patch, tgt_patch = self.my_test_segmentation_transforms(input_patch, tgt_patch)
            
#         return input_patch, tgt_patch     
# train_dataset=YourCustomDataset('Dataset/Image/test_labelisation/', '/','masks/', True)
# #27 images dans le train set


# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)



import torch
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
from torchvision.transforms.functional import to_pil_image


path='Dataset/Image/test_labelisation/'
img=Image.open(path+'228_Layer00012_Visible_MeltingEnd.jpg')
data_transform = transforms.Compose([
    #to resize
    transforms.Resize(size=(928,928)),
    #Data Augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    
    transforms.ToTensor()
])

# img_trans=data_transform(img)
# #go from [C,H,W] to [H,W,C]
# img_trans=img_trans.permute(1 , 2 , 0)
# print(img_trans.shape)




class SegmentationDataset(Dataset):
    def __init__(self, data_path, mask_path, need_transform=True):
        super().__init__()
        self.data_path = data_path
        self.mask_path = mask_path
        self.need_transform=need_transform


    def transforms(self, data, mask):
        data_transform = transforms.Compose([
        #to resize
        transforms.Resize(size=(928,928)),
        #Data Augmentation
        transforms.RandomHorizontalFlip(p=0.5)])

        return data_transform(data), data_transform(mask)            
    def __len__(self):
        l1 = os.listdir(self.data_path) 
        l2 = os.listdir(self.mask_path) 
        number_files_inp = len(l1)
        number_files_tgt = len(l2)
        if number_files_inp == number_files_tgt:
            return number_files_inp

    def __getitem__(self, idx):
        """
        Necessary function that loads and returns a sample from the dataset at a given index. 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        
        Based on the index,it identifies the input and target images location on the disk, 
        reads both items as a numpy array (float32). If the train_transform argument is True, 
        the above defined train transformations are applied. Else, the test transformations are applied
        Args:
            idx (iterable): 
        Returns:
            tensors: input and target image
        """

        l1 = os.listdir(self.data_path)
        img_name=l1[idx]

        img_data=Image.open(self.data_path+img_name)
        img_mask=Image.open(self.mask_path+img_name)
        trans=transforms.ToTensor()
        data=trans(img_data)
        mask=trans(img_mask)

            
        if self.need_transform:
            data, mask = self.transforms(data, mask)
            
        return data, mask    

training_data=SegmentationDataset(path+'data/',path+'masks/')

train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

#show an example:
k=10

img_data_show=train_features[k]
img_mask_show=train_labels[k]
to_pil_image(img_data_show).show()
to_pil_image(img_mask_show).show()


