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

def load_dataset_resize(dataset_name:str, width: int = 320, height: int = 180) -> ArrayLike:
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
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    datasets = np.zeros((len(files), width * height))

    for idx, file in enumerate(files):
        image_path = os.path.join(dataset_path, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            resized_image = cv2.resize(image, (width, height))
            datasets[idx] = resized_image.flatten()
        else:
            print(f"Warning: Unable to read image {image_path}")
    return datasets