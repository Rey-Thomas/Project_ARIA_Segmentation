import os 
import numpy as np
from PIL import Image

path = 'D:/ARIA/Dataset/Mask_parfait/'

for file in os.listdir(path):
    mask = Image.open(path+file)
    data = np.asarray(mask)
    ##print(data)
    data_inverse = ~data#np.logical_not(data)
    #print(data_inverse)
    im = Image.fromarray(data_inverse)
    im.save('D:/ARIA/Dataset/Mask_inverse/'+file)