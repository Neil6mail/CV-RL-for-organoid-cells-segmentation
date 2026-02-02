import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = "images/data1_image_300.png"  # Chemin vers l'image
# Charger l'image en niveaux de gris
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Appliquer un flou pour homogénéiser
blurred = cv2.GaussianBlur(img, (1, 1), 0)

# Seuillage inversé : le croissant devient blanc, fond noir
_, mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

# Fermer les petits trous
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
masque = cv2.medianBlur(mask, 5)
cleaned=cv2.bitwise_and(img, img, mask=masque)
# Afficher le masque

# Optionnel : retirer les petits objets
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
sizes = stats[1:, -1]  # Ignore background
min_size = 5000

# Créer nouveau masque
filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
for i in range(1, num_labels):
    if sizes[i - 1] > min_size:
        filtered_mask[labels == i] = 255


# Créer une image RGB pour coloriser
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Créer un fond bleu
blue_bg = np.full_like(img_rgb, (255, 128, 0))  # BGR pour un bleu profond

# Appliquer le masque
croissant = cv2.bitwise_and(img_rgb, img_rgb, mask=filtered_mask)
background = cv2.bitwise_and(blue_bg, blue_bg, mask=cv2.bitwise_not(filtered_mask))

# Combiner les deux
result = cv2.add(croissant, background)

plt.imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Croissant extrait avec fond coloré")
plt.show()

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, image 1
plt.imshow(img, cmap='gray')
plt.title("Image 1")
plt.axis('off')

plt.subplot(1, 2, 2)  # image 2
plt.imshow(cleaned, cmap='gray')
plt.title("Image 2")
plt.axis('off')

plt.tight_layout()
plt.show()



