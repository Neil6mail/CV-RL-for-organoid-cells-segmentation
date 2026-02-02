from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from parameters import *

print("initializing load medsam...")

### --  Load the model configuration -- ###
if Activate_Medsam:
  model_config = SamConfig.from_pretrained("flaviagiammarino/medsam-vit-base")
  processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

  # Create an instance of the model architecture with the loaded configuration
  model = SamModel(config=model_config)

  # set the device to cuda if available, otherwise use cpu
  #Update the model by loading the weights from saved file.
  if torch.cuda.is_available():
    device = "cuda"
    if finetuned_version:
      model.load_state_dict(torch.load(model_path))
  else:
    device = "cpu"
    if finetuned_version:
      model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

  model.to(device)
  print("model loaded")

### --- Evaluate the image --- ###

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

def medsam_predict(image,mask):
# get box prompt based on ground truth segmentation map

  ground_truth_mask = np.array(mask)
  prompt = get_bounding_box(ground_truth_mask)

  # prepare image + box prompt for the model
  inputs = processor(image, input_boxes=[[prompt]], return_tensors="pt")
  #inputs = processor(image, return_tensors="pt")

  # Move the input tensor to the GPU if it's not already there
  inputs = {k: v.to(device) for k, v in inputs.items()}

  model.eval()

  # forward pass
  with torch.no_grad():
      outputs = model(**inputs, multimask_output=False)

  # apply sigmoid
  medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
  # convert soft mask to hard mask
  medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
  medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)

  finetuned_medsam_seg=np.copy(medsam_seg)
  finetuned_medsam_seg_prob=np.copy(medsam_seg_prob)

  return finetuned_medsam_seg_prob, finetuned_medsam_seg


