import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchvision.transforms.functional import to_pil_image
import os

class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list, 
                 augment=False,
                 transform=None
                 ):

        self.inputs = inputs
        self.targets = targets
        self.augment=augment
        self.transform =  transform
        self.inputs_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        # Load input and target
        x, y = Image.open(input_ID), Image.open(target_ID)

        # Preprocessing
        if self.augment:
            x= self.transform(x)
            y= self.transform(y)

        # Typecasting
        trans=transforms.ToTensor()
        x, y = trans(x), trans(y)
        return x, y

path= 'D:/ARIA/Dataset/Image/'
# path='C:/Users/thoma/Desktop/ARIA/Dataset/Image/'
#targets = [path+'Photo_recadree_tout_nicolas_matlab/train/'+f for f in os.listdir(path+'Photo_recadree_tout_nicolas_matlab/train/')]

targets = [path+'Mask_parfait/few_masks/'+f for f in os.listdir(path+'Mask_parfait/few_masks/')]
#inputs = [path+'Photo_recadree/'+f for f in os.listdir(path+'Photo_recadree/')] 
inputs= [path+'photo_recadree/'+f[-37:] for f in targets]
print(f'Nombre d input: {len(inputs)}')
print(f'Nombre de masks: {len(targets)}')


transform=transforms.Compose([
        #grayscale
        transforms.Grayscale(num_output_channels=1),
        #to resize
        transforms.Resize(size=(928,928)),
        #Data Augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        
        transforms.RandomRotation(90)
        ])

training_dataset = SegmentationDataSet(inputs=inputs,
                                       targets=targets,
                                       augment=True, #Augmentation de donnée ou pas
                                       transform=transform)

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=4,
                                      shuffle=True)
train_features, train_labels = next(iter(training_dataloader))

print('Creation of the dataset over')




class SegmentationModel(pl.LightningModule):

    def __init__(self, arch, encoder_name,  in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        #print(batch)
        #print(batch[0].shape)
        #print(batch[1].shape)
        #nb_batch=len(batch)
        
        image = batch[0]#batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(), lr=0.0001) AVANT
        #return torch.optim.Adam(self.parameters(), lr=0.001) UN PEU APRES
        #return torch.optim.Adam(self.parameters(), lr=0.01)
        return torch.optim.Adam(self.parameters(), lr=0.005)

###Training the model
arch="Unet"
encoder_name="resnet34"
in_channels=3
out_classes=1
max_epochs=200


TRAIN_MODEL=False
if TRAIN_MODEL:
    model = SegmentationModel(arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)

    trainer = pl.Trainer(
        gpus=1, 
        max_epochs=max_epochs,
        default_root_dir="Models/9_Model/checkpoints/"
    )

    trainer.fit(
        model, 
        train_dataloaders=training_dataloader, 
    )

    trainer.save_checkpoint(f"Models/9_Model/final_{max_epochs}_epochs_{arch}_{encoder_name}_in_channels_{in_channels}_out_classes_{out_classes}.ckpt")


###Loading the model
if not TRAIN_MODEL:
    #model = SegmentationModel("FPN", "resnet34", in_channels=3, out_classes=1)
    #model = SegmentationModel.load_from_checkpoint(f'Models/9_Model/final_{max_epochs}_epochs_{arch}_{encoder_name}_in_channels_{in_channels}_out_classes_{out_classes}.ckpt',arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)
    model = SegmentationModel.load_from_checkpoint(f'Models/4_Model/final_200_epochs_Unet_resnet34_in_channels_3_out_classes_1.ckpt',arch=arch, encoder_name=encoder_name, in_channels=in_channels, out_classes=out_classes)

    trans=transforms.Compose([
            #grayscale
            transforms.Grayscale(num_output_channels=1),
            #to resize
            transforms.Resize(size=(928,928)),
            transforms.ToTensor()])
    path_image=path+'Photo_recadree/'  
    path_mask=path+'Mask_parfait/test/'
    path_saved=path+f'Result/{max_epochs}_epochs_{arch}_{encoder_name}_in_channels_{in_channels}_out_classes_{out_classes}/'
    try: 
        os.mkdir(path_saved)    
        print("Directory created")
    except OSError as error:
        print("Directory can not be created")

    images=os.listdir(path_mask)#['228_Layer00300_Visible_MeltingEnd.jpg','228_Layer00120_Visible_MeltingEnd.jpg','228_Layer00471_Visible_MeltingEnd.jpg','269_Layer00030_Visible_MeltingEnd.jpg']
    for elt in images:
        filename=elt[-37:]
        image_test=trans(Image.open(path_image+filename))


        with torch.no_grad():
            model.eval()
            logits = model(image_test)
            pr_masks = logits.sigmoid()
            transform = transforms.ToPILImage()      
            pr_masks=transform(pr_masks.squeeze())
            pr_masks.save(path_saved+filename[:-4]+'_result'+filename[-4:])
            image_test=transform(image_test)
            image_test.save(path_saved+filename)
            image_mask=Image.open(path_mask+filename)
            image_mask.save(path_saved+filename[:-4]+'_mask'+filename[-4:])


            # to_pil_image(image_test).show()
            # to_pil_image(pr_masks.squeeze()).show()