from cProfile import run
from cgi import test
from torch import nn
import torch
from torch.utils import data
from PIL import Image
from torchvision import datasets, transforms
import os
import numpy as np
import pandas as pd
from utils_fct import DownConvulution, UpConvulution
import torch.optim as optim
import random as rand
import time
import json
from torchmetrics.classification import JaccardIndex, Accuracy

### DataLoader doit créer une liste de tensor (les 4 qui sont demandé dans l'architecture (A/B/C/D)) 
#les images seront d'abord resize en 928 par 928

class SegmentationDataSet_3legs(data.Dataset):
    def __init__(self,
                datas: list,  #[[str(nb_build),str(nb_layer),str(tile_nb)],...]
                path_data= 'D:/ARIA/Dataset/Data_3_legs/', #on attends un str contenant le path vers le dossier Data_3_legs (plus tard séparer le test du trainset)
                tile_size = 200,
                augment=False,
                transform=None
                ):
        self.datas = datas
        self.path_data = path_data
        self.tile_size = tile_size
        self.augment=augment
        self.transform =  transform
        self.inputs_dtype = torch.float32

    def __len__(self):
        return len(self.datas)

    def __getitem__(self,
                    index: int):
        # Select the sample
        #directory_build_layer = self.files[index]
        data_image = self.datas[index]
        directory_build_layer = data_image[0]+'_Layer'+data_image[1]
        file = os.listdir(self.path_data+directory_build_layer)

        for elt in file:
            if 'Layering' in elt:
                layering_image = Image.open(self.path_data+directory_build_layer+'/'+elt)
            if 'Melting' in elt:
                melting_image = Image.open(self.path_data+directory_build_layer+'/'+elt)


        # Typecasting
        trans=transforms.Compose([
              #to resize
              transforms.ToTensor()
              ])

        melting_image, layering_image= trans(melting_image), trans(layering_image)

        A=torch.cat((layering_image,melting_image),0)

        path_tile=self.path_data+directory_build_layer+'/'+f'tile_size={self.tile_size}/{data_image[2]}/'
        for elt in os.listdir(path_tile):
            if 'layering' in elt:
                crop_layering_image = Image.open(path_tile+elt)
            if 'melting' in elt:
                crop_melting_image = Image.open(path_tile+elt)

            if 'mask' in elt and 'inverse' not in elt:
                crop_mask_image = Image.open(path_tile+elt)
                if data_image[0]=='269': # une erreur dans la création du dataset nécessite cette ligne pour etre sur de prendre la bonne photo
                    crop_mask_image = Image.open(path_tile+'crop_mask.jpg')
            
            if 'inverse' in elt:
                crop_mask_inverse_image = Image.open(path_tile+elt)

            if 'localisation_x' in elt:
                df = pd.read_csv(path_tile+elt)
                df=np.array(df)
                tensor_coord_x=torch.tensor(df,dtype=torch.float)

            if 'localisation_y' in elt:
                df = pd.read_csv(path_tile+elt)
                df=np.array(df)
                tensor_coord_y=torch.tensor(df,dtype=torch.float)

        if list(crop_mask_image.getdata()) == list(crop_mask_inverse_image.getdata()):
            print('ERROOOORRRRRRRRRRRRRRRRRRRRRRRRR')

        
        crop_layering_image, crop_melting_image, crop_mask_image, crop_mask_inverse_image = trans(crop_layering_image), trans(crop_melting_image), trans(crop_mask_image), trans(crop_mask_inverse_image)
        B=torch.cat((crop_layering_image,crop_melting_image),0)
        D=torch.cat((tensor_coord_x.unsqueeze(0), tensor_coord_y.unsqueeze(0)),0)
        D.type=float
        crop_mask_concatenate = torch.cat((crop_mask_image,crop_mask_inverse_image),0)

        # print(f'A shape {A.size()}')
        # print(f'B shape {B.size()}')
        # print(f'D shape {D.size()}')
        # print(f'mask shape {crop_mask_concatenate.size()}')

        return A,B,D,crop_mask_concatenate



### Architecture du modèle

class three_legs(torch.nn.Module):
    def __init__(self,tile_size,nb_classes,n_cam):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.tile_size = tile_size
        self.nb_classes = nb_classes
        self.n_cam = n_cam
        self.leg_A_1 = nn.Sequential(
                                   nn.Conv2d(in_channels=2*self.n_cam, out_channels=32, kernel_size = 5, stride=1), #padding de 2 reflect car symétrique vérifier que c'est bien la bonne méthode car je suis pas sur de comprendre vraiment le symétrique dans l'article
                                   nn.ReLU())

        self.leg_A_2 = nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
                                   nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 5, stride=1, padding = 2, padding_mode = 'reflect'),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2),
                                   nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 5, stride=1, padding = 2, padding_mode = 'reflect'),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2,stride=2),
                                   nn.Flatten(),
                                   nn.Linear(2048,256),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(256,16),
                                   nn.Dropout(p=0.5),
                                   nn.Unflatten(1, (16, 1, 1))

                                   )

        self.leg_B = nn.Sequential(nn.Conv2d(in_channels=2*self.n_cam, out_channels=128, kernel_size = 11, stride=1, padding = 5, padding_mode = 'reflect'), #padding de 2 reflect car symétrique vérifier que c'est bien la bonne méthode car je suis pas sur de comprendre vraiment le symétrique dans l'article
                                   nn.ReLU()
                                   )

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.DownSample1 = DownConvulution(2*self.n_cam,32)
        self.DownSample2 = DownConvulution(32,64)
        self.DownSample3 = DownConvulution(64,128)
        self.DownSample4 = DownConvulution(128,256)
        self.DownSample5 = DownConvulution(256,512)
        self.C_intermediate = nn.Sequential(nn.Flatten(),nn.Linear(8192,512),nn.Dropout(p=0.5),nn.Unflatten(1, (512, 1, 1)))
        self.Upsample1 = nn.Sequential(nn.Upsample(scale_factor=8, mode='nearest'),nn.Conv2d(in_channels=512, out_channels=512, kernel_size = 1, stride=1),nn.ReLU())
        self.Upsample2 = UpConvulution(1024,[512,256])
        self.Upsample3 = UpConvulution(512,[256,128])
        self.Upsample4 = UpConvulution(256,[128,64])
        self.Upsample5 = UpConvulution(128,[64,32])
        self.Upsample6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size = 3, stride=1),nn.ReLU(),nn.Upsample(size=self.tile_size, mode='bilinear'))

        self.classification = nn.Sequential(nn.Conv2d(178,self.nb_classes, kernel_size = 1, stride=1),
                                            nn.Softmax(dim=self.nb_classes))


    def forward(self, liste_entry):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        for elt in ['A','B','C','D']:
            if elt == 'A':
                #print(elt)
                A=liste_entry[0]
                #print(A.size())
                #A=nn.functional.interpolate(A, size=(32,32), mode='bilinear')
                A=nn.functional.interpolate(A, scale_factor=1/29, mode='bilinear')
                #print(A.size())
                A=nn.functional.pad(A, pad=(2,2,2,2), mode='reflect')
                #print(A.size())
                A=self.leg_A_1(A)
                A=transforms.functional.crop(A, top=2, left=2, height=32, width=32) #crop
                #print(A.size())
                A=self.leg_A_2(A)
                #print(A.size())
                A=nn.functional.interpolate(A, size=(self.tile_size,self.tile_size), mode='bilinear')
                #print(A.size())
            if elt== 'B':
                #print(elt)
                B=liste_entry[1]
                #print(B.size())
                B=self.leg_B(B)
                #print(B.size())
            if elt == 'C':
                #print(elt)
                C=liste_entry[1]
                #print(C.size())
                C=nn.functional.interpolate(C, size=(128,128), mode='bilinear')
                #print(C.size())
                C=self.DownSample1(C)
                C_32=C
                C=self.maxpool(C)
                #print(C.size())
                C=self.DownSample2(C)
                C_64=C
                C=self.maxpool(C)
                #print(C.size())
                C=self.DownSample3(C)
                C_128=C
                C=self.maxpool(C)
                #print(C.size())
                C=self.DownSample4(C)
                C_256=C
                C=self.maxpool(C)
                #print(C.size())
                C=self.DownSample5(C)
                C_512=C
                C=self.maxpool(C)
                #print(C.size())
                C=self.C_intermediate(C)
                #print(C.size())
                C=self.Upsample1(C)
                #print(C.size())
                #print(C_512.size())
                C=torch.cat((C,C_512),1)
                #print(C.size())
                C=self.Upsample2(C)
                #print(C.size())
                C=torch.cat((C,C_256),1)
                C=self.Upsample3(C)
                #print(C.size())
                C=torch.cat((C,C_128),1)
                C=self.Upsample4(C)
                #print(C.size())
                C=torch.cat((C,C_64),1)
                C=self.Upsample5(C)
                #print(C.size())
                C=torch.cat((C,C_32),1)
                C=self.Upsample6(C)
                #print(C.size())
            if elt == 'D':
                #print(elt)
                D=liste_entry[2]
                #print(D.size())

        # print(f'A: {A.type()}')
        # print(f'B: {B.type()}')
        # print(f'C: {C.type()}')
        # print(f'D: {D.type()}')
        #print('result')
        result = torch.cat((B,A,C,D),1)
        #print(result.size())
        result=self.classification(result)
        #print(result.size())
        return result


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.05)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.1)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.004)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)
    tile_size=200
    nb_classes=2
    n_cam=1
    lr=10**(-3)
    batch_size=32
    nb_epochs = 3


    liste_data=[]
    #dict_nb_layer = {'213': 447, '225': 747, '228': 553, '234': 415, '238': 998, '239': 873, '240': 873, '265': 748, '269': 373, '421': 423, '428': 323, '437': 198, '474': 748, '481': 467, '488': 1182, '496': 91, '507': 317, '514': 1840}
    dict_nb_layer = {'213': 447, '228': 553, '234': 415, '238': 998, '239': 873, '240': 873, '265': 748, '269': 373, '421': 423, '428': 323, '437': 198, '474': 748, '481': 467, '488': 1182, '496': 91, '507': 317, '514': 1840}
    non_valid = [['437','00041'],['269','00003'],['239','00223'],['213','00190'],['213','00277'],['213','00301'],['213','00388'],['225','00629'],['225','00468'],['225','00485'],['225','00550'],['225','00558'],['225','00614'],['225','00632'],['225','00638'],['238','00398'],['238','00586'],['238','00599'],['238','00759'],['238','00928'],['265','00467'],['265','00661'],['488','00999']]

    for key in dict_nb_layer.keys():
        nb_build=key
        for nb_layer in range(17,dict_nb_layer[key]):
            for nb_tile in range(25):
                if nb_layer<10:
                    layer=f'0000{nb_layer}'
                if nb_layer<100:
                    layer=f'000{nb_layer}'
                if 99<nb_layer<1000:
                    layer=f'00{nb_layer}'
                if 999<nb_layer<10000:
                    layer=f'0{nb_layer}'
                if [nb_build,layer] not in non_valid :
                    liste_data.append([nb_build,layer,str(nb_tile)])
        
    print(len(liste_data))

    # training_dataset=SegmentationDataSet_3legs([['228','00045','2'],['228','00047','6'],['428','00045','2'],['428','00047','6'],['428','00145','2'],['228','00247','6'],['428','00300','2'],['428','00147','6'],
    # ['228','00245','16'],['228','00047','16'],['428','00045','12'],['428','00047','16'],['428','00145','22'],['228','00247','16'],['428','00300','2'],['428','00147','16'],
    # ['428','00245','16'],['428','00047','16'],['228','00045','12'],['228','00047','16'],['228','00145','22'],['428','00247','16'],['228','00300','12'],['228','00152','16']])

    print(liste_data[0])
    rand.seed(42)
    rand.shuffle(liste_data)
    print(type(liste_data))
    print(liste_data[0])
    training_liste_data = liste_data[:int(len(liste_data)*0.7)]
    validation_liste_data = liste_data[int(len(liste_data)*0.7):int(len(liste_data)*0.8)]
    testing_liste_data = liste_data[int(len(liste_data)*0.8):len(liste_data)+1]

    training_dataset=SegmentationDataSet_3legs(training_liste_data)
    valid_dataset=SegmentationDataSet_3legs(validation_liste_data)
    test_dataset=SegmentationDataSet_3legs(testing_liste_data)

    print(len(training_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))


    training_dataloader = data.DataLoader(dataset=training_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)
    valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)
    test_dataloader = data.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)


    model=three_legs(tile_size=tile_size,nb_classes=nb_classes,n_cam=n_cam)
    model.to(device)
    model.apply(initialize_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr,eps=10**(-4))
    jaccard = JaccardIndex(task="multiclass", num_classes=2).to(device)
    accuracy = Accuracy(task="multiclass").to(device)
    

    loss_dict = {'epochs' : nb_epochs, 'dataset': dict_nb_layer, 'train' : [], 'val' : [], 'iou melt':[],'iou powder':[],'acc melt':[],'acc powder':[]}

    PATH = f'Model_full_training/model_3legs_tile_{tile_size}_nb_classes_{nb_classes}_lr_{lr}_batch_size_{batch_size}_epochs_{nb_epochs}/'
    os.mkdir(PATH)
    start = time.time()
    print(nb_epochs)
    for epochs in range(nb_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                mean_loss = 0
                for batch_idx, (A, B, D, labels) in enumerate(training_dataloader):
                    A, B, D, labels = A.to(device), B.to(device), D.to(device), labels.to(device)
                    optimizer.zero_grad()
                    output = model([A,B,D])
                    #_ , pred = torch.max(output,1)
                    loss = criterion(output, labels)
                    mean_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % batch_size*100 == 0:
                        print(f'Epochs {epochs} [{batch_idx * len(A)}/{len(training_dataloader.dataset)} ({100. * batch_idx / len(training_dataloader):.0f}%)]: loss = {loss.item():.6f}')
                loss_dict[phase].append(mean_loss/len(training_dataloader)) #14 batchs trouver comment trouver le nombre exact tout le temps


            if phase == 'val':
                model.eval()
                mean_loss = 0
                mean_iou_melt = 0
                mean_iou_powder = 0
                mean_accuracy_melt = 0
                mean_accuracy_powder = 0
                for batch_idx, (A, B, D, labels) in enumerate(valid_dataloader):
                    A, B, D, labels = A.to(device), B.to(device), D.to(device), labels.to(device)
                    output = model([A,B,D])
                    loss = criterion(output, labels)
                    mean_loss += loss.item()
                    # print('pouet 1')
                    # print(output.shape)
                    # print('pouet 2 ')
                    # print(output.to(torch.int).shape)
                    # print('pouet 3 ')
                    # print(output[:,0].shape)
                    # print('pouet 4 ')
                    #print(100*output[0,0])
                    #print(100*output[0,1])

                    mean_iou_melt += jaccard(100*output[:,0], labels[:,0].to(torch.int))
                    mean_iou_powder += jaccard(100*output[:,1], labels[:,1].to(torch.int))
                    mean_accuracy_melt += accuracy(100*output[:,0], labels[:,0].to(torch.int))
                    mean_accuracy_powder += accuracy(100*output[:,1], labels[:,1].to(torch.int))
                loss_dict[phase].append(mean_loss/len(valid_dataloader))   
                loss_dict['iou melt'].append(float(mean_iou_melt/len(valid_dataloader)))
                loss_dict['iou powder'].append(float(mean_iou_powder/len(valid_dataloader)))
                loss_dict['acc melt'].append(float(mean_accuracy_melt/len(valid_dataloader)))
                loss_dict['acc powder'].append(float(mean_accuracy_powder/len(valid_dataloader)))

        print(f'epoch nb {epochs}: train loss = { loss_dict["train"][-1]}, valid loss = { loss_dict["val"][-1] }, IOU score melt= {loss_dict["iou melt"][-1]}, IOU score powder= {loss_dict["iou powder"][-1]}, Acc score melt= {loss_dict["acc melt"][-1]}, Acc score powder= {loss_dict["acc powder"][-1]}')

        
        
        #model_file = PATH + '/model_epochs_' + str(epochs) +f'_train_loss_{loss_dict["train"][-1]}_valid_loss_{loss_dict["val"][-1]}_IOU_score_melt_{loss_dict["iou melt"][-1]}_IOU_score_powder_{loss_dict["iou powder"][-1]}_acc_score_melt_{loss_dict["acc melt"][-1]}acc_score_powder_{loss_dict["acc powder"][-1]}.pth'
        model_file = PATH + '/model_epochs_' + str(epochs)+'.pth'
        torch.save(model.state_dict(), model_file)
    finish = time.time()



    #PATH = f'Model/MultiScale_loss_{loss_name}_lr_{lr}_epochs_{nb_epochs}_batch_size_{batch_size}_use_weight_{use_weight}.pth'
    #torch.save(model, PATH)
    
    minutes, seconds = divmod(finish-start , 60)
    loss_dict['training time'] = f'{minutes} min'
    loss_dict['name'] = PATH[5:]
    loss_dict['lr'] = lr
    
    #print(weight.to('cpu').numpy())
    #loss_dict['weight'] = list(weight.to('cpu').numpy()) #,not serialiazble

    with open(PATH +'/trainning_info.json', 'w') as jsonfile:
        # Reading from json file
        json.dump(loss_dict, jsonfile)