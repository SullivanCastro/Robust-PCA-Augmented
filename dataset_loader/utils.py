import os
import numpy as np
from numpy.typing import ArrayLike
import cv2

DATASET_PATH = "/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented/Datasets/"


def load_dataset(dataset_name:str, width: int = 320, height: int = 180) -> ArrayLike:
    """
    Load the dataset from the path

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    width : int
        Width of the image by default 320   
    height : int
        Height of the image by default 180

    Returns
    -------
    np.array
        Dataset
    """
    datasets = np.zeros(shape=(len(os.listdir(os.path.join(DATASET_PATH, dataset_name))), width*height))
    for raw, _, files in os.walk(os.path.join(DATASET_PATH, dataset_name)):
        for idx, file in enumerate(files):
            if file.endswith('.jpg'):
                image = cv2.imread(os.path.join(raw, file), cv2.IMREAD_GRAYSCALE)
                datasets[idx] = image.flatten()
    return datasets