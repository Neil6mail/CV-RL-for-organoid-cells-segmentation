import tifffile
import cv2
import os

# Chemins d'accès
images_path = "data/data1/D1-1M-sample1-hydrogel.tif"
masks_path = "data/data1/D1-1M-sample1-spheroid.tif"
output_folder = "images"
index = 200  # Index de l'image/mask à extraire

# Extraire le nom du dossier parent (data1 → i=1)
data_i = os.path.basename(os.path.dirname(images_path))  # ex: 'data1'

# Préfixe pour le nom de fichier
prefix = f"{data_i}_"

# Fichiers de sortie
image_filename = f"{prefix}image_{index}.png"
mask_filename = f"{prefix}mask_{index}.png"

# Charger les stacks
large_images = tifffile.imread(images_path)
large_masks = tifffile.imread(masks_path)

# Extraire l'image et le masque à l'index donné
image_i = large_images[index]
mask_i = large_masks[index]

# Créer le dossier de sortie si nécessaire
os.makedirs(output_folder, exist_ok=True)

# Fonction de normalisation en uint8
def normalize_to_uint8(img):
    if img.dtype == 'uint8':
        return img
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

# Normaliser et sauvegarder
image_i = normalize_to_uint8(image_i)
mask_i = normalize_to_uint8(mask_i)

cv2.imwrite(os.path.join(output_folder, image_filename), image_i)
cv2.imwrite(os.path.join(output_folder, mask_filename), mask_i)

print(f"[✔] Image sauvegardée dans {output_folder}/{image_filename}")
print(f"[✔] Masque sauvegardé dans {output_folder}/{mask_filename}")
