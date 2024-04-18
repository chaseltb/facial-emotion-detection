import os
import cv2
import numpy as np
from multiprocessing import Pool

DATA_DIR: str = "data"
"""Relative path to the data directory."""
NEW_DATA_DIR: str = "processed_data"
"""Relative path to the process data directory."""
IMAGE_EXT: list[str] = [".jpg", ".jpeg", ".png"]
"""List of image extensions to use for fetching images from data directory."""
IMG_SIZE: int = 96
"""Default size to resize images to."""

# Placed at the top for easy manipulation
def pipline(img: np.ndarray) -> np.ndarray:
    """
    Preprocesses the image using a series of functions.
    This is the function that would be called when processing images in real time from a camera feed.
    :param img: The image to process. This image will be fed through the pipeline.
    :return: The final processed image after being fed through all processing functions.
    """

    # Only resizing during preprocessing at the moment
    img = resize(img)

    return img

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """ Converts an image to grayscale. """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """ Resizes image to a square image of size x size. """
    return cv2.resize(img, (size, size))

def write_image(item: tuple[str, np.ndarray]) -> None:
    """
    Writes an image to a file. Using this function for multiprocessing.
    :param item: Tuple containing the path to the image and the image itself.
    """

    path, img = item
    cv2.imwrite(path, img)

def main():
    """The main function to read training images from data directory, preprocess them,
    and save them into a new data directory to later be used for training."""

    # Ingest all files from data directory
    files: list[str] = []  # List of full image paths from the data directory
    for root, _, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXT:
                files.append(os.path.join(os.getcwd(), root, filename))

    # Read and process images from files
    with Pool() as pool:
        images: list[np.ndarray] = pool.map(cv2.imread, files)
        images = pool.map(pipline, images)

    # Copy old directory structure to new directory
    for path in files:
        new_path = path.replace(DATA_DIR, NEW_DATA_DIR)
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Write images to new directory
    new_paths: list[str] = [path.replace(DATA_DIR, NEW_DATA_DIR) for path in files]
    with Pool() as pool:
        pool.map(write_image, zip(new_paths, images))


if __name__ == "__main__":
    main()
