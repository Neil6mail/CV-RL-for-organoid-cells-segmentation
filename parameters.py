### -- Parameters -- ###
from computer_vision_algo import *


image_path = "images/data1_image_300.png" 
mask_path = "images/data1_mask_300.png"  

# image_path2 = "images/unsharp_image.png" 
# image_path3 = "images/combined_image.png" 

# image_path = "images/cni.png" 
# mask_path = "images/cni.png"  


# image_path = "image_bw.png"




patch_size = 256
step = 128

Activate_Medsam=False

model_path = "model_checkpoints/model_checkpoint_1_100_epochs.pth"
finetuned_version = True #True if you want to use the finetuned version of the model, False if you want to use the pretrained version.


ALGO_REGISTRY=ALGO_REGISTRY_30

#à donner à medsam en entrainement:
#une ou la couleur est bien contrastée
#une ou les cellules sont bien pleines

#une ou les contours sont bien visibles
