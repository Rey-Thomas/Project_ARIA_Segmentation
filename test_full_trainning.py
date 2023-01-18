from torch import nn
import torch
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from semantic_seg_v4 import SegmentationDataSet_3legs, three_legs
from utils_fct import supperposition_mask_on_image,image_creation, get_concat_v_blank
from PIL import Image
import random as rand
from torchmetrics.classification import JaccardIndex, Accuracy
import json


#PATH = 'Model_full_training/model_3legs_tile_200_nb_classes_2_lr_0.001_batch_size_32_epochs_1/'
PATH = 'model_3legs/model_3legs_tile_200_nb_classes_2_lr_0.0001_epochs_20_batch_size_64_nb_mini_batch_20/'
PATH='Model_full_training/model_3legs_tile_200_nb_classes_2_lr_0.01_batch_size_32_epochs_5/'

batch_size=24
model_name = 'model_epochs_4.pth'
confiance = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = three_legs(200,2,1)
model.load_state_dict(torch.load(PATH+model_name))
# model=torch.load(PATH+model_name)
model.to(device)
print('model  eval')
model.eval()


# liste_data=[]    
# dict_nb_layer = {'213': 447, '228': 553, '234': 415, '238': 998, '239': 873, '240': 873, '265': 748, '269': 373, '421': 423, '428': 323, '437': 198, '474': 748, '481': 467, '488': 1182, '496': 91, '507': 317, '514': 1840}

# non_valid = [['437','00041'],['269','00003'],['239','00223'],['213','00190'],['213','00277'],['213','00301'],['213','00388'],['225','00629'],['225','00468'],['225','00485'],['225','00550'],['225','00558'],['225','00614'],['225','00632'],['225','00638'],['238','00398'],['238','00586'],['238','00599'],['238','00759'],['238','00928'],['265','00467'],['265','00661'],['488','00999']]

# for key in dict_nb_layer.keys():
#     nb_build=key
#     for nb_layer in range(17,dict_nb_layer[key]):
#         for nb_tile in range(25):
#             if nb_layer<10:
#                 layer=f'0000{nb_layer}'
#             if nb_layer<100:
#                 layer=f'000{nb_layer}'
#             if 99<nb_layer<1000:
#                 layer=f'00{nb_layer}'
#             if 999<nb_layer<10000:
#                 layer=f'0{nb_layer}'
#             if [nb_build,layer] not in non_valid :
#                 liste_data.append([nb_build,layer,str(nb_tile)])

# rand.seed(42)
# rand.shuffle(liste_data)
# print(type(liste_data))
# print(liste_data[0])
# training_liste_data = liste_data[:int(len(liste_data)*0.7)]
# validation_liste_data = liste_data[int(len(liste_data)*0.7):int(len(liste_data)*0.8)]
# testing_liste_data = liste_data[int(len(liste_data)*0.8):len(liste_data)+1]

# test_dataset=SegmentationDataSet_3legs(testing_liste_data)

# test_dataloader = data.DataLoader(dataset=test_dataset,
#                                         batch_size=batch_size,
#                                         shuffle=False)
 

# jaccard = JaccardIndex(task="multiclass", num_classes=2).to(device)
# accuracy = Accuracy(task="multiclass").to(device)

# test_dict = {}

# mean_iou_melt = 0
# mean_iou_melt_0_2 = 0
# mean_iou_melt_0_3 = 0
# mean_iou_melt_0_4 = 0
# mean_iou_melt_0_6 = 0
# mean_iou_melt_0_7 = 0
# mean_iou_melt_0_8 = 0
# mean_iou_melt_0_9 = 0
# mean_iou_melt_1 = 0


# mean_iou_powder = 0
# mean_iou_powder_1 = 0

# mean_accuracy_melt = 0
# mean_accuracy_powder = 0
# mean_acc_melt_0_2 = 0
# mean_acc_melt_0_3 = 0
# mean_acc_melt_0_4 = 0
# mean_acc_melt_0_6 = 0
# mean_acc_melt_0_7 = 0
# mean_acc_melt_0_8 = 0
# mean_acc_melt_0_9 = 0
# mean_acc_melt_1 = 0
# mean_acc_powder_1 = 0

# for batch_idx, (A, B, D, labels) in enumerate(test_dataloader):
#     A, B, D, labels = A.to(device), B.to(device), D.to(device), labels.to(device)
#     output = model([A,B,D])
#     mean_iou_melt += jaccard(100*output[:,0], labels[:,0].to(torch.int))
#     mean_iou_powder += jaccard(100*output[:,1], labels[:,1].to(torch.int))  
#     mean_accuracy_melt += accuracy(100*output[:,0], labels[:,0].to(torch.int))
#     mean_accuracy_powder += accuracy(100*output[:,1], labels[:,1].to(torch.int))
#     for confiance in [0.2,0.3,0.4,0.6,0.7,0.8,0.9,1]:
#         mask_conf=(100*output>confiance).float()
#         if confiance == 0.2:
#             mean_iou_melt_0_2 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_2 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 0.3:
#             mean_iou_melt_0_3 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_3 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 0.4:
#             mean_iou_melt_0_4 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_4 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 0.6:
#             mean_iou_melt_0_6 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_6 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 0.7:
#             mean_iou_melt_0_7 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_7 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 0.8:
#             mean_iou_melt_0_8 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_8 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 0.9:
#             mean_iou_melt_0_9 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_0_9 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#         if confiance == 1:
#             mean_iou_melt_1 += jaccard(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_acc_melt_1 += accuracy(mask_conf[:,0], labels[:,0].to(torch.int))
#             mean_iou_powder_1 += jaccard(mask_conf[:,1], labels[:,1].to(torch.int))
#             mean_acc_powder_1 += accuracy(mask_conf[:,1], labels[:,1].to(torch.int))

#     if batch_idx % batch_size*100 == 0:
#                         print(f' [{batch_idx * len(A)}/{len(test_dataloader.dataset)} ({100. * batch_idx / len(test_dataloader):.0f}%)]')
# test_dict['iou melt'] = float(mean_iou_melt/len(test_dataloader))
# test_dict['iou melt 0.2'] = float(mean_iou_melt_0_2/len(test_dataloader))
# test_dict['iou melt 0.3'] = float(mean_iou_melt_0_3/len(test_dataloader))
# test_dict['iou melt 0.4'] = float(mean_iou_melt_0_4/len(test_dataloader))
# test_dict['iou melt 0.6'] = float(mean_iou_melt_0_6/len(test_dataloader))
# test_dict['iou melt 0.7'] = float(mean_iou_melt_0_7/len(test_dataloader))
# test_dict['iou melt 0.8'] = float(mean_iou_melt_0_8/len(test_dataloader))
# test_dict['iou melt 0.9'] = float(mean_iou_melt_0_9/len(test_dataloader))
# test_dict['iou melt 1'] = float(mean_iou_melt_1/len(test_dataloader))
# test_dict['iou powder 1'] = float(mean_iou_powder_1/len(test_dataloader))
# test_dict['iou powder'] = float(mean_iou_powder/len(test_dataloader))
# test_dict['acc melt'] = float(mean_accuracy_melt/len(test_dataloader))
# test_dict['acc powder'] = float(mean_accuracy_powder/len(test_dataloader))
# test_dict['acc melt 0.2'] = float(mean_acc_melt_0_2/len(test_dataloader))
# test_dict['acc melt 0.3'] = float(mean_acc_melt_0_3/len(test_dataloader))
# test_dict['acc melt 0.4'] = float(mean_acc_melt_0_4/len(test_dataloader))
# test_dict['acc melt 0.6'] = float(mean_acc_melt_0_6/len(test_dataloader))
# test_dict['acc melt 0.7'] = float(mean_acc_melt_0_7/len(test_dataloader))
# test_dict['acc melt 0.8'] = float(mean_acc_melt_0_8/len(test_dataloader))
# test_dict['acc melt 1'] = float(mean_acc_melt_1/len(test_dataloader))
# test_dict['acc powder 1'] = float(mean_acc_powder_1/len(test_dataloader))

# with open(PATH +'/test_result.json', 'w') as jsonfile:
#         # Reading from json file
#         json.dump(test_dict, jsonfile)

test_dataset_list = []
for image_nb in [['225','00100'],['239','00223'],['269','00003'],['437','00041']]:
    for tile in range(25):
        test_dataset_list.append([image_nb[0],image_nb[1],str(tile)])
    testset=SegmentationDataSet_3legs(test_dataset_list)

    testset_dataloader = data.DataLoader(dataset=testset,
                                      batch_size=1,
                                      shuffle=False)
    for batch_idx, (A, B, D, labels) in enumerate(testset_dataloader):
        A, B, D, labels = A.to(device), B.to(device), D.to(device), labels.to(device)
        output=model([A,B,D])
        output=100*output
        image_temp = supperposition_mask_on_image(output, data_image = test_dataset_list[batch_idx],confiance=confiance)
        if batch_idx in [4,9,14,19,24]:
            image_temp = image_temp.crop((71,0,200,200))
        if batch_idx not in [0,5,10,15,20] :
            if batch_idx in [1,2,3,4]:
                image_1 = image_creation(image_1, image_temp, str(batch_idx))
            if batch_idx in [6,7,8,9]:
                image_2 = image_creation(image_2, image_temp, str(batch_idx))
            if batch_idx in [11,12,13,14]:
                image_3 = image_creation(image_3, image_temp, str(batch_idx))
            if batch_idx in [16,17,18,19]:
                image_4 = image_creation(image_4, image_temp,str(batch_idx))
            if batch_idx in [21,22,23,24]:
                image_5 = image_creation(image_5, image_temp, str(batch_idx))

        if batch_idx == 0 :
            image_1 = image_temp
        if batch_idx == 5 :
            image_2 = image_temp
        if batch_idx == 10 :
            image_tot = image_creation(image_1, image_2, str(batch_idx))
            image_3 = image_temp
        if batch_idx == 15 :
            image_tot = image_creation(image_tot, image_3,str(batch_idx))
            image_4 = image_temp
        if batch_idx == 20 :
            image_tot = image_creation(image_tot, image_4, str(batch_idx))
            image_5 = image_temp

        if batch_idx == 24:
            image_5 = image_5.crop((0,71,1000,200))
            image_tot = get_concat_v_blank(image_tot, image_5)
            image_tot = image_tot.crop((0,0,929,929))
            image_tot.save(PATH+f'{image_nb[0]}_{image_nb[1]}_confiance_{confiance}.png')
            image_tot= Image.new('RGB',[0,0],(0,0,0))

    test_dataset_list = []
