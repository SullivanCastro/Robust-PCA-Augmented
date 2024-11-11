import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.append("/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented")
from model_custom_pca.rpca import Robust_PCA

def load_dataset(path:str) -> np.array:
    """
    Load the dataset from the path

    Parameters
    ----------
    path : str
        Path to the dataset

    Returns
    -------
    np.array
        Dataset
    """
    datasets = np.zeros(shape=(len(os.listdir(path)), 320*180))
    for raw, dir, files in os.walk(path):
        for idx, file in enumerate(files):
            if file.endswith('.jpg'):
                image = cv2.imread(os.path.join(raw, file), cv2.IMREAD_GRAYSCALE)
                datasets[idx] = image.flatten()
    return datasets


def normalize_image(image: np.array) -> np.array:
    """
    Normalize the image

    Parameters
    ----------
    image : np.array
        Image to normalize

    Returns
    -------
    np.array
        Normalized image
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

dataset_cyprien = load_dataset("/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented/Datasets/City")
print(dataset_cyprien.shape)


rpca = Robust_PCA(dataset_cyprien)
L, S = rpca.fit()
lr_image = normalize_image(L.mean(axis=0)).astype(np.uint8).reshape((180, 320)).astype(np.uint8)
lr_image = Image.fromarray(lr_image, mode="L")

s_image = normalize_image(S.mean(axis=0)).astype(np.uint8).reshape((180, 320)).astype(np.uint8)
s_image = Image.fromarray(s_image, mode="L")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

ax[0, 0].imshow(lr_image, cmap="gray")
ax[0, 0].set_title("Low-rank matrix")

ax[0, 1].imshow(S.mean(axis=0).reshape((180, 320)), cmap="gray")
ax[0, 1].set_title("Sparse matrix")

plt.show()
