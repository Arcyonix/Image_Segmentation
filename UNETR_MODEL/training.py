#!/usr/bin/env python
# coding: utf-8

# This Main jupyter file is where the training of the main models occur. This is done with references to the MONAI tutorial on 3d_segmentation
# 
#     1)Transforms for dictionary format data.
#     2)Define a new transform according to MONAI transform API.
#     3)Load NRRD image with metadata, load a list of images and stack them.
#     4)Randomly adjust intensity for data augmentation.
#     5)Cache IO and transforms to accelerate training and validation.
#     6)3D UNETR model, DiceCE loss function, Mean Dice metric and HausdorffDistanceMetric
#     7)Saves all the loss graph plot and dice plot
#     8)Saves plot of final predicted label vs baseline label
#     
#     
# 
# 
# [1]: Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H.R. and Xu, D., 2022. Unetr: Transformers for 3d medical image segmentation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 574-584).

# In[19]:


import os
import shutil
import tempfile
import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

import nrrd

import numpy as np
import SimpleITK as sitk

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    Resized,
    SaveImage,
)
print("A")
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric

from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
print_config()


# ## Setup data directory (For saving and loading of data)
# ## If not specified a temporary directory will be used.

# In[2]:


directory = '/home/katan/'
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# ## Setup Transformer
# 

# In[3]:


train_transforms = Compose(
    [

        # Loads images and corresponding labels from the specified keys.
        LoadImaged(keys=["image", "label"]),
                         
        #Ensures that the channel dimension is the first dimension in both the image and label arrays.
        EnsureChannelFirstd(keys=["image", "label"]),
        #Reshapes the image and label arrays according to the provided orientation codes ("RAS" in this case),
        #ensuring consistent orientation across different images.
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        #’RAS’ represents 3D orientation: (Left, Right), (Posterior, Anterior), (Inferior, Superior).
          Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.0),  # Adjusted for the new volume sizes
            mode=("bilinear", "nearest"),
        ),
        
        #Scales the intensity values of the images within a specified range (a_min to a_max) to a new range (b_min to b_max). 
        #It's commonly used for normalization.
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        #Crops the foreground (non-zero intensity regions) of the images and corresponding labels.
        #It uses the source key "image" to determine the foreground region.
        #good when the valid part is small in the image
        CropForegroundd(keys=["image", "label"], source_key="image"),


        #Performs random cropping based on positive and negative labels. 
        #It selects random spatial locations based on the specified spatial size and the number of positive and negative samples.
                         
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        #Randomly flips the images and labels along the specified spatial axis with a given probability.
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        # Randomly rotates the images and labels by 90 degrees multiples, with a specified probability (prob) and maximum number of rotations (max_k=3).
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        # Randomly shifts the intensity values of the images within a specified offset range (offsets) with a given probability.
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
      Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 0),  # Adjusted for the new volume sizes
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)


# # Set up data directory

# In[4]:


data_dir = r'C:\Users\jefft\TDT4265_StarterCode_2024\Project\Dataset'
split_json = "dataset_0.json"

datasets = data_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=32,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=8, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


# # Check data shape and visualize
# 

# In[5]:


slice_map = {
    "Diseased_1.nrrd": 45,
    "Diseased_2.nrrd": 23,
    "Diseased_17.nrrd": 112,
    "Normal_1.nrrd": 79,
    "Normal_4.nrrd": 23,
    "Normal_6.nrrd": 64,
    
}
case_num = 0

#Validation Set
img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
img_shape = img.shape
label_shape = label.shape
print(f"image shape: {img_shape}, label shape: {label_shape}")
plt.figure("image", (18, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(img[0, :, :, slice_map[img_name]].detach().cpu(), cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[0, :, :, slice_map[img_name]].detach().cpu())
# Save the plot
plt.savefig('plot0_25k.png') 
plt.show()



# # Create Model, Loss, Optimizer
# 
# 

# In[6]:


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if torch.cuda.is_available():
    print("CUDA is available. You can use GPU acceleration.")
else:
    print("CUDA is not available. GPU acceleration is not possible.")

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


# # Training Functions

# In[7]:


# Create an instance of HausdorffDistanceMetric
#hausdorff_metric = HausdorffDistanceMetric()


def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
             # Calculate Hausdorff distance
            #hausdorff_metric(val_output_convert, val_labels_convert)
            
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
        # Get Hausdorff distance
        #hausdorff_distance = hausdorff_metric.aggregate().item()
        #hausdorff_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(  # noqa: B038
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            #hausdorff_distance_values.append(hausdorff_distance)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


# # PyTorch training process
# 

# In[8]:


max_iterations = 25000 #change this accordingly
eval_num = 250 #change this accordingly
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
hausdorff_distance_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))


# In[ ]:


print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


# # Plot the loss and metric
# 

# In[ ]:


plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("Iteration")
plt.plot(x, y)

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Save the plot
plt.savefig('plot_25k.png') 

plt.show()


# # Check best model output with the input image and label
# 

# In[26]:


case_num = 0
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]])
    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Save the plot
    plt.savefig('plot2_25k.png')  

    plt.show()


# In[32]:


case_num = 0
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
    img = val_ds[case_num]["image"]
    label = val_ds[case_num]["label"]
    val_inputs = torch.unsqueeze(img, 1).cuda()
    val_labels = torch.unsqueeze(label, 1).cuda()
    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)

    # Stack slices along the specified dimension
    output_volume = torch.argmax(val_outputs, dim=1).detach().cpu()

    # Save the 3D volume as an image
    output_volume_np = output_volume.numpy()
    output_volume_np = np.moveaxis(output_volume_np, 0, -1)  # Move channel dimension to the last
    output_volume_np = np.squeeze(output_volume_np)  # Remove singleton channel dimension if present

    # Save the 3D volume as an image file
    output_volume_img = sitk.GetImageFromArray(output_volume_np)
    sitk.WriteImage(output_volume_img, 'output_volume_25k.nii')

    # Plot the first slice for visualization
    plt.imshow(output_volume_np[:, :, slice_map[img_name]])
    plt.title("Output Volume")
    plt.colorbar()
    plt.show()

