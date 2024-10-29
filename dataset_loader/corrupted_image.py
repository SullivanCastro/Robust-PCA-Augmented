from utils import load_dataset, DATASET_PATH
import numpy as np
import os


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
        self.dataset_name = dataset_name
        self.dataset_original = load_dataset(os.path.join(DATASET_PATH, dataset_name))
        self.width, self.height = width, height

    
    def _missing_pixels(self, missing_rate: float = 0.05) -> None:
        """
        Create a corrupted dataset with missing pixels

        Parameters
        ----------
        missing_rate : float, optional
            Rate of missing pixels, by default 0.5
        """
        self.dataset_missing = self.dataset_original.copy()
        for i in range(self.dataset_missing.shape[0]):
            missing_pixels = np.random.choice(self.dataset_missing.shape[1], int(missing_rate * self.dataset_missing.shape[1]), replace=False)
            self.dataset_missing[i, missing_pixels] = 0

        return self.dataset_missing

    
    def _occlusion(self, occlusion_rate: float = 0.001) -> None:
        """
        Create a corrupted dataset with occlusion patches

        Parameters
        ----------
        occlusion_rate : float, optional
            Rate of occlusion, by default 0.1%
        """
        self.dataset_occlusion = self.dataset_original.copy()
        patch_size = int(occlusion_rate * self.dataset_occlusion.shape[1])
        for i in range(self.dataset_occlusion.shape[0]):
            first_pixel = np.random.randint(0, self.dataset_occlusion.shape[1] - patch_size - patch_size * self.width)
            for row in range(patch_size):
                self.dataset_occlusion[i, first_pixel+row*self.width:first_pixel+row*self.width+patch_size] = 0
        
        return self.dataset_occlusion


    def load_corrupted_dataset(self, type: str = 'missing'):
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
        return dataset_corrupted


if __name__ == "__main__":
    loader = Corrupted_Dataset_Loader("Cyprien")
    dataset_missing = loader.load_corrupted_dataset()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(dataset_missing[0].reshape(loader.height, loader.width), cmap="gray")
    plt.show()