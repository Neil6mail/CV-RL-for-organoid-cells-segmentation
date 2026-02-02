import numpy as np
from PIL import Image
from patchify import patchify  #Only to handle large images
from parameters import *
if Activate_Medsam:
    from load_medsam import *
from metric import *
from computer_vision_algo import *


print("initializing image treatment...")

def read_image(image_path,mask_path):
    """
    This function reads the image and mask from the given paths.
    """
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Convert to numpy arrays
    image_array = np.array(image)  # Convert PIL Image to NumPy array
    mask_array = np.array(mask)    # Convert PIL Image to NumPy array

    return image_array, mask_array

def init_image(image_path, mask_path, patch_size = 256, step = 128):
    
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    #Desired patch size for smaller images and step size.

    image_array = np.array(image)  # Convert PIL Image to NumPy array
    mask_array = np.array(mask)    # Convert PIL Image to NumPy array

    patches_img = patchify(image_array, (patch_size, patch_size), step=step)  # Step=128 for 128 patches means there will be overlap
    patches_mask = patchify(mask_array, (patch_size, patch_size), step=step)  # Step=128 for 128 patches means there will be overlap    

    all_img_patches = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):

            single_patch_img = patches_img[i,j,:,:]
            all_img_patches.append(single_patch_img)
    images = np.array(all_img_patches)
    

    all_mask_patches = []
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):

            single_patch_mask = patches_mask[i,j,:,:]
            all_mask_patches.append(single_patch_mask)
    masks = np.array(all_mask_patches)

    #look for the best patch with the most organoid in it.
    max_i=0
    for i in range(len(masks)):
        if np.count_nonzero(masks[i][10:-10][10:-10]) > np.count_nonzero(masks[max_i][10:-10][10:-10]):
            max_i=i
    
    patch_image = images[max_i]
    patch_mask = masks[max_i]

    return patch_image, patch_mask

def pre_medsam(im1,im2=False,im3=False): #can take 1,2 or 3 images as input. If only one image is given, it will be used for all three channels.

    if type(im3)!=bool:im3=im1
    if type(im2)!=bool:im2=im1

        # Conversion en uint8 si nécessaire
    im1 = (im1 * 255).astype(np.uint8) if im1.dtype != np.uint8 else im1
    im2 = (im2 * 255).astype(np.uint8) if im2.dtype != np.uint8 else im2
    im3 = (im3 * 255).astype(np.uint8) if im3.dtype != np.uint8 else im3

    # Convert the image to RGB by stacking it three times

    test_image = np.stack((im1,im2,im3), axis=-1)  # Stack to create RGB. 
    test_image = Image.fromarray(test_image)  # Convert back to PIL Image
    
    return test_image

def evaluate(images, mask):
    evaluation = []
    score = []
    for i in range(len(images)):
        image=images[i]

        # Preprocess the image for Medsam
        stacked_img = pre_medsam(image,image,image)

        # Predict using Medsam
        probability_map,prediction = medsam_predict(stacked_img, mask)  

        #compute the F1 score
        score.append(compute_f1_score(prediction, mask))

        evaluation.append((probability_map, prediction))
        
        print(f"Evaluation {i+1}/{len(images)} done. Score is {score[i]}")  # Use score[i] for individual score

    return evaluation, score

def apply_computer_vision_algo(image, algo_sequence=list( ALGO_REGISTRY.keys() ), sequence=False ) :
    images = []

    # Loop for each algorithm in the sequence

    for i in range(len(algo_sequence)):
        algo_name = str(algo_sequence[i])  # Ensure algo_name is a string

        func = ALGO_REGISTRY[algo_name]["func"]
        parameters = ALGO_REGISTRY[algo_name]["params"]

        new_image = func(image, **parameters)
        # Apply the algorithm to the image
        images.append(new_image)

        # Save the processed images
        
        name="processed"
        for e in algo_sequence[i]:
            name+=f"{e}" 
        name+=".png"

        if not sequence:
            path=f"transformation_image/individual/{name}"

        cv2.imwrite(path, new_image)
        
    print("[✔] Images sauvegardée dans transformation_image")
    return images

def apply_computer_vision_sequence(image, algo_sequence):

    
    images = []

    # Loop through the algorithm sequence

    for i in range(len(algo_sequence)):
        algo_name = algo_sequence[i]

        func = ALGO_REGISTRY[algo_name]["func"]
        parameters = ALGO_REGISTRY[algo_name]["params"]

        # Apply the algorithm to the image
        images.append(func(image, **parameters))
        image = images[-1]

        # Save the processed images
        
        name="processed"
        for e in algo_sequence[:i+1]:
            name+=f"_{e}_" 
        name+=".png"
        
        cv2.imwrite(f"transformation_image/{name}", image)
    print("[✔] Images sauvegardée dans transformation_image")
    return images

def clean_background(image, threshold=60):


    # Appliquer un flou pour homogénéiser
    blurred = cv2.GaussianBlur(image, (1, 1), 0)

    # Seuillage inversé : le croissant devient blanc, fond noir
    _, clean_background = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # Fermer les petits trous
    kernel = np.ones((5, 5), np.uint8)
    clean_background = cv2.morphologyEx(clean_background, cv2.MORPH_CLOSE, kernel)
    masque = cv2.medianBlur(clean_background, 5)
    cleaned = cv2.bitwise_and(image, image, mask=masque)
    return cleaned
