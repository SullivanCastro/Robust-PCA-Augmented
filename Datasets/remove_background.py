import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
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
    datasets = np.zeros(shape=(len(os.listdir(path)), 720*1280))
    for raw, dir, files in os.walk(path):
        for idx, file in enumerate(files):
            if file.endswith('.jpg'):
                image = cv2.imread(os.path.join(raw, file), cv2.IMREAD_GRAYSCALE)
                # plt.figure()
                # plt.imshow(Image.fromarray(image, mode="L"), cmap="gray")
                # plt.show()
                datasets[idx] = image.flatten()
    return datasets

dataset_cyprien = load_dataset("/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented/Datasets/Cyprien")
print(dataset_cyprien.shape)


rpca = Robust_PCA(dataset_cyprien)
L, S = rpca.fit()
image = L[0].reshape((720, 1280)).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.imshow(Image.fromarray(image, mode="L"), cmap="gray")
plt.title("Low-rank matrix")