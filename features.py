from image_treatment import *
from computer_vision_algo import *
from show_computer_vision_algo import *
if Activate_Medsam:
    from load_medsam import *
from metric import *

print("initializing features...")




### - Features - ###

def show_all_algo(algo_sequence,image_path, mask_path, patch_size=256, step=128,human_eval=True):
    """
    This function applies all algorithms in the algo_sequence to the image and displays the results.
    """
     # Load the image and mask
    image, mask = init_image(image_path, mask_path, patch_size, step)

    # Apply the selected algorithms to the image. If algo_sequence is absent, all algorithms will be applied in the order of the dictionary.
    images = apply_computer_vision_algo(image, algo_sequence)

    if human_eval:
        # Display the images transformed by the algorithms in the sequence and save the human scores to a CSV file
        viewer = AlgoViewer_human(images, algo_sequence)
        plt.show()
        export_scores_to_csv()

    return images, mask

def eval_all_algo_individualy(algo_sequence,image_path, mask_path, patch_size=256, step=128,human_eval=True):
    """
    This function applies all algorithms in the algo_sequence to the image and displays the results AND evaluates them with MedSAM.
    """
     # Load the image and mask
    image, mask = init_image(image_path, mask_path, patch_size, step)

    # Apply the selected algorithms to the image. If algo_sequence is absent, all algorithms will be applied in the order of the dictionary.
    images = apply_computer_vision_algo(image, algo_sequence)

    # Evaluate the transformations with Medsam and compute_f1_score
    evaluation, score = evaluate(images, mask)  # use of medsam and compute_f1_score

    # Display the images transformed by the algorithms and the predictions by Medsam, and save MedSAM scores to a csv file.
    
    viewer = AlgoViewer_medsam(images, evaluation, mask, names=algo_sequence, score=score)
    plt.show()

    # Save the images and evaluation results
    export_scores_to_csv(mode="medsam", scores=score)

def all_algo_list(Registry=ALGO_REGISTRY):
    """
    This function returns a list of all available algorithms.
    """
    return list(Registry.keys())

def eval_all_algo_sequence(algo_sequence,image_path, mask_path, patch_size=256, step=128,human_eval=True):
    #unfinished function to apply a sequence of algorithms to the image and display the results.
    cv2.imwrite("images/final_images.png", images[-1])
    cv2.imwrite("images/evaluation_image.png", evaluation)

    #algo_sequence=rl_sequencer() to be coded
    pass

def eval_images(images_paths, mask_path ,name_csv="last_test.csv", patch_size=256, step=128):

    good_format_images = []

    for image_path in images_paths : 
        image, mask = init_image(image_path, mask_path, patch_size, step)
        good_format_images.append(image)

    evaluation, score = evaluate(good_format_images, mask)  # use of medsam and compute_f1_score
    
    viewer = AlgoViewer_medsam(good_format_images, evaluation, mask, score=score)
    plt.show()

        # Save the images and evaluation results
    export_scores_to_csv(mode="image", name_csv=name_csv, scores=score)
    

    pass

def view_images(images,algo_sequence):

    """
    This function displays the images transformed by the algorithms in the sequence.
    """
    
    viewer = AlgoViewer_human(images, algo_sequence)
    plt.show()

def explore_algo(image_path, mask_path):
    """
    This function applies all algorithms in the algo_sequence to the image and displays the results.
    """
    # Load the image and mask
    image, mask = init_image(image_path, mask_path)
    
    viewer = TkAlgoEditor(image,mask)
    plt.show()

    return image