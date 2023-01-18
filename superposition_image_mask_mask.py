from PIL import Image
import os

#path='C:/Users/thoma/Desktop/ARIA/Dataset/Image/'
path='D:/ARIA/'
#targets = [path+'Photo_recadree_tout_nicolas_matlab/train/'+f for f in os.listdir(path+'Photo_recadree_tout_nicolas_matlab/train/')]

inputs = [path+'ARIA/Mask_bbx_228_269_428/'+f for f in os.listdir('Mask_bbx_228_269_428/')]
targets= [path+'Dataset/Image/Photo_recadree_tout_nicolas_matlab/'+f[-37:] for f in inputs]

print(len(targets))
print(len(inputs))

for i in range(len(targets)):
    # Opening the primary image (used in background)
    img1 = Image.open(inputs[i])
    #img1 = img1.convert("RGBA")
    
    # Opening the secondary image (overlay image)
    img2 = Image.open(targets[i])
    
    img2 = img2.convert("RGBA")
    img2.putalpha(32)
    d = img2.getdata()
    new_image = []
    for item in d:
    
        # change all white (also shades of whites)
        # pixels to yellow
        if item[0] in list(range(200, 256)):
            new_image.append((255, 0, 0))
        else:
            new_image.append(item)
            
    # update image data
    img2.putdata(new_image)
    img2.save('Mask_sliced/Superposition_pitch=0.1_Mask_parfait_mask_Matlab/'+inputs[i][-37:-4]+'.png')
    
    # Pasting img2 image on top of img1 
    # starting at coordinates (0, 0)
    img1.paste(img2, (0,0), mask = img2)
    
    
    # Displaying the image
    #img1.save(path+'Photo_recadree_tout_nicolas_matlab/Superposition/'+inputs[i][-37:])
    img1.save('Mask_sliced/Superposition_pitch=0.1_Mask_parfait_mask_Matlab/'+inputs[i][-37:-4]+'.png')