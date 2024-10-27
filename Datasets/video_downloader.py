from pytube import YouTube
import os

URL = "https://www.youtube.com/watch?v=WvhYuDvH17I&ab_channel=masterryze"
PATH = "/Users/sullivancastro/Desktop/MVA/Geometric Data Analysis/Robust-PCA-Augmented/Datasets/Videos"

def downloadYoutube(vid_url, path):
    yt = YouTube(vid_url)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)

    yt.download(path)

downloadYoutube(URL, PATH)

