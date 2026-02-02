from features import *
from parameters import *

print("initializing main...")

#you have to activate conda env to use medsam ; see requirements

#to load a sequence, load applied_algo.csv files. You will find them in the presentation folder

#when saving a sequence, the image, the stats and the list of algo are saved in a folder created in the selected folder.
explore_algo(image_path, mask_path)


#select a sequence of algorithms to apply to the image.
algo_sequence = ["gaussian_blur", "unsharp_mask", "adaptive_threshold", "otsu_threshold", "canny"]
#algo_sequence = all_algo_list() #list of all available algorithms 


#if you want to use medsam, Activate_Medsam=True in parameter.py
eval_all_algo_individualy(algo_sequence,image_path, mask_path,human_eval=True)


#evaluate images with MedSAM and compute_f1_score
images_paths = [image_path]
eval_images(images_paths, mask_path, name_csv="test.csv")
