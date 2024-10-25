# Convert image to np.array of images all the 0.2 seconds
import cv2
from PIL import Image
import numpy as np

def save_images_from_video(path: str, image: np.array):
    """
    Save images from video

    Parameters
    ----------
    path : str
        Path to the video
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image.save(path)
    

def read_videos(path, time_interval=1):
    """
    Read videos from path

    Parameters
    ----------
    path : str
        Path to the videos

    Returns
    -------
    np.array
        Array of images
    """
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_numb = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_numb % (time_interval * fps) == 0:
            save_images_from_video(f"/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented/Datasets/Cyprien/Cyprien_{frame_numb}.jpg", frame)
    cap.release()
    return np.array(frames)

read_videos(f"/Users/sullivancastro/Downloads/Cyprien_Video.mp4")