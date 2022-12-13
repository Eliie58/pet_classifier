import os
from skimage.io import imread
from train import resize_image
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO)


def predict(model: Pipeline, image_path: str) -> str:
    """
    Predict the class of the provided image

    Parameter
    ---------
    model: Pipeline
        The model to use for predicting
    image_path: str
        The path of the image to predict

    Return
    ------
    str
        The class of the image
    """
    # Read the image from disk
    im = imread(image_path)

    # Resize the image
    height = width = 64
    im_resized = resize_image(im, width, height)
    logging.info(model.predict([im_resized]))
