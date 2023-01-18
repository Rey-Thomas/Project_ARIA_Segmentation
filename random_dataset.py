from PIL import Image
import os, os.path
import random

#taille du dataset à changer selon le besoin

nb_image=1248   
path = "D:/ARIA/Dataset/Image/Mask_parfait/"

random_numbers = random.sample(range(nb_image), round(0.8*nb_image)) #80% des images seront dans le trainset

filename_train=[os.listdir(path)[i] for i in random_numbers]


for f in os.listdir(path):
    if 'MeltingEnd' in f:
        ext = os.path.splitext(f)[1]
        if f in filename_train:
            img=Image.open(os.path.join(path,f))
            img.save('D:/ARIA/Dataset/Image/Mask_parfait/train/'+f)
        else:
            img=Image.open(os.path.join(path,f))
            img.save('D:/ARIA/Dataset/Image/Mask_parfait/test/'+f)
