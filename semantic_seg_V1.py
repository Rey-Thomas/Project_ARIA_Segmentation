from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

img = read_image("C:/Users/thoma/Desktop/INESC TEC/dataset/dataset/34_Color.png")

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()
#print(preprocess)


# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
print(normalized_masks.size())
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
#print(class_to_idx)
#mask = normalized_masks[0]#[0, class_to_idx["person"]]
#to_pil_image(mask).show()
