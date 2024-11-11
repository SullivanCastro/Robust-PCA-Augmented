from utils import load_dataset, DATASET_PATH
import numpy as np
import os
import json
import re
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import cv2
# import svd
from scipy.linalg import svd


CORRUPTED_DATASET_PATH = f"C:\\MVA\\1er Semestre\\G Data Analysis\\RPCA\\Robust-PCA-Augmented\\Corrupted_Datasets"

class Corrupted_Dataset_Loader:
    """
    Class to load the corrupted dataset
    """

    def __init__(self, dataset_name: str, width: int = 320, height: int = 180) -> None:
        """
        Initialize the corrupted dataset loader

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        width : int, optional
            Width of the image, by default 320
        height : int, optional
            Height of the image, by default 180
        """
        self._dataset_name = dataset_name
        self._dataset_original = load_dataset(os.path.join(DATASET_PATH, dataset_name))
        self._dataset_corrupted = self._dataset_original.copy()
        self._width, self._height = width, height
        self._json = {}

    
    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns
        -------
        int
            Length of the dataset
        """
        return self._dataset_corrupted.shape[0]
    

    @property
    def shape(self) -> tuple:
        """
        Get the shape of the dataset

        Returns
        -------
        tuple
            Shape of the dataset
        """
        return self._width, self._height


    def __getitem__(self, index: int) -> np.array:
        """
        Get the image at the index

        Parameters
        ----------
        index : int
            Index of the image

        Returns
        -------
        np.array
            Image
        """
        if len(self._json) > 0:
            return self._dataset_corrupted[index], self._json[index][0], self._json[index][1]
        else:
            return self._dataset_corrupted[index]

    
    def _missing_pixels(self, missing_rate: float = 0.05) -> None:
        """
        Create a corrupted dataset with missing pixels

        Parameters
        ----------
        missing_rate : float, optional
            Rate of missing pixels, by default 0.5
        """
        height_image = self._dataset_corrupted.shape[1]
        for i in range(self._dataset_corrupted.shape[0]):
            missing_pixels = np.random.choice(height_image, int(missing_rate * height_image), replace=False)
            self._dataset_corrupted[i, missing_pixels] = 0

        return self._dataset_corrupted
    

    def _occlusion(self, occlusion_rate: float = 0.001, save=False) -> None:
        """
        Create a corrupted dataset with occlusion patches

        Parameters
        ----------
        occlusion_rate : float, optional
            Rate of occlusion, by default 0.1%
        """
        patch_size = int(occlusion_rate * self._dataset_corrupted.shape[1])
        for i in range(self._dataset_corrupted.shape[0]):
            first_pixel = np.random.randint(0, self._dataset_corrupted.shape[1] - patch_size - patch_size * self._width)
            for row in range(patch_size):
                self._dataset_corrupted[i, first_pixel+row*self._width:first_pixel+row*self._width+patch_size] = 0
            self._json[i] = ((first_pixel % loader._width, first_pixel // loader._width), patch_size)
        
        return self._dataset_corrupted


    def _save_corrupted_dataset(self, type) -> None:
        """
        Save the corrupted dataset

        Parameters
        ----------
        type : str
            Type of corruption
        
        Returns
        -------
        None
        """
        output_path = os.path.join(CORRUPTED_DATASET_PATH, type, self._dataset_name)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the corrupted dataset
        for i in range(self._dataset_corrupted.shape[0]):
            cv2.imwrite(os.path.join(output_path, f"{self._dataset_name}_{i}.jpg"), self._dataset_corrupted[i].reshape(self._height, self._width))
                                   
        if len(self._json) > 0:
            json.dump(self._json, open(os.path.join(output_path, f"annotation.json"), "w"))


    def load_corrupted_dataset(self, type: str = 'missing', save=False) -> np.array:
        """
        Load the corrupted dataset

        Parameters
        ----------
        type : str, optional
            Type of corruption, by default 'missing'

        Returns
        -------
        np.array
            Corrupted dataset
        """
        if type == 'missing':
            dataset_corrupted = self._missing_pixels()
        elif type == 'occlusion':
            dataset_corrupted = self._occlusion()
        else:
            raise ValueError("The type of corruption is not recognized")

        if save:
            self._save_corrupted_dataset(type)
        
        return dataset_corrupted


if __name__ == "__main__":

    loader = Corrupted_Dataset_Loader("Cyprien")
    dataset_missing = loader.load_corrupted_dataset(type='occlusion', save=True)

    # Plot the first image of the corrupted dataset
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(dataset_missing[0].reshape(*loader.shape[::-1]), cmap="gray")

    # Add the occlusion patch witht the annotation file
    from matplotlib.patches import Rectangle
    annotation_dataset = json.load(open(os.path.join(CORRUPTED_DATASET_PATH, "occlusion", "Cyprien", "annotation.json"), "r"))
    image_first_pixel, image_path_size = annotation_dataset["0"]
    plt.gca().add_patch(Rectangle((image_first_pixel[0], image_first_pixel[1]), image_path_size, image_path_size, linewidth=1, edgecolor='r', facecolor='none'))

    plt.show()

    # base_path = "C:\\MVA\\1er Semestre\\G Data Analysis\\RPCA\\att_faces\\"
    # num_people = 5
    # num_images_per_person = 5
    # X, y = load_images(base_path, num_people, num_images_per_person)
    # loader = Corrupted_Dataset_Loader("Cyprien")
    # loader._dataset_original = X
    # loader._dataset_corrupted = loader._dataset_original.copy()
    # dataset_missing = loader.load_corrupted_dataset(type='occlusion', save=True)
    # X = dataset_missing