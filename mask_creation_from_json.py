import json
import cv2
import os
import numpy as np
from glob import glob
    
    
output_dir = "Dataset/Image/test_labelisation/masks/"
#creating the ground_truth folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
    
json_files = glob("Dataset/Image/test_labelisation/*.json")
#loading the json file
for image_json in json_files:
    with open(image_json) as file:
        data = json.load(file)
    filename = data["imagePath"][-37:-4]
    
    # creating a new ground truth image
    mask = np.zeros((data["imageHeight"], data["imageWidth"]), dtype='uint8')
    for shape in data['shapes']:
        mask = cv2.fillPoly(mask, [np.array(shape['points'], dtype=np.int32)], 255)

    # saving the ground truth masks
    cv2.imwrite(os.path.join(output_dir,filename) + ".jpg", mask)
