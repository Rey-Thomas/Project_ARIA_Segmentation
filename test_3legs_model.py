from torch import nn
import torch
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from semantic_seg_v4 import SegmentationDataSet_3legs, three_legs
from utils_fct import supperposition_mask_on_image


tile_size=200
nb_classes=2
n_cam=1
epochs=200
lr=10**(-4)
batch_size=32
nb_mini_batch = 10
confiance = 0.009
criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
PATH = f'model_3legs/model_3legs_tile_{tile_size}_nb_classes_{nb_classes}_lr_{lr}_epochs_{epochs}_batch_size_{batch_size}_nb_mini_batch_{nb_mini_batch}_use_of_0.005_separator.pth'
PATH = 'model_3legs/CrossEntropyLoss/model_3legs_tile_200_nb_classes_2_epochs_200_lr_001.pth'
#PATH = 'Model_full_training/model_3legs_tile_200_nb_classes_2_lr_0.001_batch_size_32_epochs_1/model_epochs_0.pth'
#PATH = 'model_3legs/model_3legs_tile_200_nb_classes_1_lr_0.0001_epochs_2_batch_size_8_nb_mini_batch_50.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device ='cpu'
model=torch.load(PATH,map_location=torch.device(device))
print('model  eval')
model.eval()


# testset=SegmentationDataSet_3legs([['269','00295','16']])
# testset_dataloader = data.DataLoader(dataset=testset,
#                                       batch_size=1,
#                                       shuffle=True)
# A, B, D, label = next(iter(testset_dataloader))
# output=model([A,B,D])
# print(label)
# print(output)
# to_pil_image(label.squeeze()[0]).show()
# to_pil_image(output.squeeze()[0]).show()
# to_pil_image(output.squeeze()[1]).show()
# mask_final=(output>confiance).float()
# print(f'loss output: {criterion(output,label)}')
# print(f'loss output 0: {criterion(output.squeeze()[0],label.squeeze()[0])}')
# print(f'loss output 1: {criterion(output.squeeze()[1],label.squeeze()[1])}')
# print(f'loss mask_final: {criterion(mask_final,label)}')
# print(f'loss mask_final 0: {criterion(mask_final.squeeze()[0],label.squeeze()[0])}')
# print(f'loss mask_final 1: {criterion(mask_final.squeeze()[1],label.squeeze()[1])}')
# to_pil_image(mask_final.squeeze()[0]).show()


# testset=SegmentationDataSet_3legs([['228','00235','5']])
# testset_dataloader = data.DataLoader(dataset=testset,
#                                       batch_size=1,
#                                       shuffle=True)
# A, B, D, label = next(iter(testset_dataloader))
# output=model([A,B,D])
# to_pil_image(label.squeeze()).show()
# to_pil_image(output.squeeze()).show()
# mask_final=(output>confiance).float()
# print(f'loss output: {criterion(output,label)}')
# print(f'loss output 0: {criterion(output.squeeze(),label.squeeze())}')
# print(f'loss mask_final: {criterion(mask_final,label)}')
# print(f'loss mask_final 0: {criterion(mask_final.squeeze(),label.squeeze())}')
# to_pil_image(mask_final.squeeze()).show()


testset=SegmentationDataSet_3legs([['228','00235','15']])
testset_dataloader = data.DataLoader(dataset=testset,
                                      batch_size=1,
                                      shuffle=True)
A, B, D, label = next(iter(testset_dataloader))
output=model([A,B,D])

supperposition_mask_on_image(output, data_image = ['228','00235','15'],confiance=0.015).show()
# to_pil_image(label.squeeze()).show()
# to_pil_image(output.squeeze()).show()
# mask_final=(output>confiance).float()
# print(f'loss output: {criterion(output,label)}')
# print(f'loss output 0: {criterion(output.squeeze(),label.squeeze())}')
# print(f'loss mask_final: {criterion(mask_final,label)}')
# print(f'loss mask_final 0: {criterion(mask_final.squeeze(),label.squeeze())}')
# to_pil_image(mask_final.squeeze()).show()

# testset=SegmentationDataSet_3legs([['228','00235','15']])
# testset_dataloader = data.DataLoader(dataset=testset,
#                                       batch_size=1,
#                                       shuffle=True)
# A, B, D, label = next(iter(testset_dataloader))
# output=model([A,B,D])

# to_pil_image(label.squeeze()[0]).show()
# to_pil_image(output.squeeze()[0]).show()
# #to_pil_image(output.squeeze()[1]).show()
# #to_pil_image(output.squeeze()[0]).show()
# mask_final=(output>confiance).float()
# print(f'loss output: {criterion(output,label)}')
# print(f'loss output 0: {criterion(output.squeeze()[0],label.squeeze()[0])}')
# print(f'loss output 1: {criterion(output.squeeze()[1],label.squeeze()[1])}')
# print(f'loss mask_final: {criterion(mask_final,label)}')
# print(f'loss mask_final 0: {criterion(mask_final.squeeze()[0],label.squeeze()[0])}')
# print(f'loss mask_final 1: {criterion(mask_final.squeeze()[1],label.squeeze()[1])}')
# to_pil_image(mask_final.squeeze()[0]).show()


# testset=SegmentationDataSet_3legs([['428','00235','16']])
# testset_dataloader = data.DataLoader(dataset=testset,
#                                       batch_size=1,
#                                       shuffle=True)
# A, B, D, label = next(iter(testset_dataloader))
# output=model([A,B,D])

# to_pil_image(label.squeeze()[0]).show()
# to_pil_image(output.squeeze()[0]).show()
# to_pil_image(output.squeeze()[1]).show()
# #to_pil_image(output.squeeze()[0]).show()
# mask_final=(output>confiance).float()
# print(f'loss output: {criterion(output,label)}')
# print(f'loss output 0: {criterion(output.squeeze()[0],label.squeeze()[0])}')
# print(f'loss output 1: {criterion(output.squeeze()[1],label.squeeze()[1])}')
# print(f'loss mask_final: {criterion(mask_final,label)}')
# print(f'loss mask_final 0: {criterion(mask_final.squeeze()[0],label.squeeze()[0])}')
# print(f'loss mask_final 1: {criterion(mask_final.squeeze()[1],label.squeeze()[1])}')
# to_pil_image(label.squeeze()[1]).show()
# to_pil_image(mask_final.squeeze()[0]).show()
# to_pil_image(mask_final.squeeze()[1]).show()

