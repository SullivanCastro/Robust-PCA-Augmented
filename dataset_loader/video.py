import numpy as np
from numpy.typing import ArrayLike
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = "/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented/Datasets/"


import sys
sys.path.append("/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented")
from model_custom_pca.rpca import Robust_PCA, Robust_PCA_augmented


class VideoLoader:

    @staticmethod
    def _save_images_from_video(path: str, image: ArrayLike) -> None:
        """
        Save images from video

        Parameters
        ----------
        path : str
            Path to the video
        image : ArrayLike
            Image

        Returns
        -------
        None
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = image.resize((320, 180), Image.LANCZOS)
        image.save(path)
    
    
    @staticmethod
    def read_videos(dataset_name: str, time_interval: float = 1) -> ArrayLike:
        """
        Read videos from path

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        time_interval : float, optional

        Returns
        -------
        np.array
            Array of images
        """
        cap = cv2.VideoCapture(os.path.join(DATASET_PATH, "Videos", f"{dataset_name}.mp4"))
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Create the dataset folder
        if dataset_name not in os.listdir(DATASET_PATH):
            os.mkdir(os.path.join(DATASET_PATH, dataset_name))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_numb = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if frame_numb % int(time_interval * fps) == 0:
                VideoLoader._save_images_from_video(os.path.join(DATASET_PATH, dataset_name, f"{dataset_name}_{frame_numb}.jpg"), frame)
        cap.release()
        return np.array(frames)


    @staticmethod
    def _load_dataset(path:str) -> ArrayLike:
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
    

    @staticmethod
    def remove_background(method, dataset_name: str, width: int = 320, height: int = 180, plot: bool = False) -> None:
        """
        Remove the background from the video

        Parameters
        ----------
        method : str
            Method to remove the background among ["rpca", "rpca_augmented"]
        dataset : ArrayLike
            Dataset
        dataset_name : str
            Name of the dataset
        width : int, optional
            Width of the image, by default 320
        height : int, optional
            Height of the image, by default 180
        plot : bool, optional
            Plot the background image, by default False
        
        Returns
        -------
        None
        """
        # Load the dataset
        dataset = VideoLoader._load_dataset(os.path.join(DATASET_PATH, dataset_name))

        # Fit the Robust PCA model
        if method == "rpca":
            rpca = Robust_PCA(dataset)
        elif method == "rpca_augmented":
            rpca = Robust_PCA_augmented(dataset)
        else:
            raise ValueError("Method not found")
        L, _ = rpca.fit()

        # Generate the mean background image
        image = L.mean(axis=0).astype(np.uint8).reshape((height, width)).astype(np.uint8)
        image = Image.fromarray(image, mode="L")

        # Save the image
        image.save(os.path.join(DATASET_PATH.replace("Datasets", ""), "Results", rpca.__class__.__name__, f"results_{dataset_name}.png"))

        if plot:
            plt.figure(figsize=(10, 5))
            plt.imshow(image, cmap="gray")
            plt.title("Low-rank matrix")
            plt.show()


if __name__ == "__main__":
    for dataset_name in ["Street"]:
        VideoLoader.read_videos(dataset_name, time_interval=0.5)
        # VideoLoader.remove_background(method= 'rpca', dataset_name=dataset_name, plot=False)
        VideoLoader.remove_background(method= 'rpca_augmented', dataset_name=dataset_name, plot=False)