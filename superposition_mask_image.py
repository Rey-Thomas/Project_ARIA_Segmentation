from PIL import Image
import os

#path='C:/Users/thoma/Desktop/ARIA/Dataset/Image/'
path='D:/ARIA/'
#targets = [path+'Photo_recadree_tout_nicolas_matlab/train/'+f for f in os.listdir(path+'Photo_recadree_tout_nicolas_matlab/train/')]
targets = ['D:/ARIA/Dataset/Mask_parfait/'+ f for f in os.listdir('D:/ARIA/Dataset/Mask_parfait/')]


inputs= [path+'Dataset/Image/Photo_recadree/'+f[-37:] for f in targets]
print(len(targets))
print(len(inputs))

for i in range(len(targets)):
    # Opening the primary image (used in background)
    img1 = Image.open(inputs[i])
    
    # Opening the secondary image (overlay image)
    img2 = Image.open(targets[i])
    img2.putalpha(64)
    
    # Pasting img2 image on top of img1 
    # starting at coordinates (0, 0)
    img1.paste(img2, (0,0), mask = img2)
    
    
    # Displaying the image
    #img1.save(path+'Photo_recadree_tout_nicolas_matlab/Superposition/'+inputs[i][-37:])
    img1.save('D:/ARIA/Dataset/Superposition_Mask/'+inputs[i][-37:])
