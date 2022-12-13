import numpy as np
import os
from tqdm import tqdm
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)


def train_model(dataset_path: str) -> Pipeline:
    """
    This function trains an image binary classification model,
    and returns it.

    Parameter
    ---------
    dataset_path: str
        The path of the dataset

    Return
    ------
    Pipeline
        The trained model in an skearn pipeline 
    """
    # Load the dataset
    dataset = load_dataset(dataset_path)
    logging.info('Number of samples: %s', len(dataset['data']))
    logging.info('Classes: %s', np.unique(dataset['label']))

    # Resize the images
    height = width = 64
    preprocessed_dataset = copy.deepcopy(dataset)
    preprocessed_dataset['data'] = resize_images(
        preprocessed_dataset['data'], width, height)

    # Extract X and y
    X = np.array(preprocessed_dataset['data'])
    y = np.array(preprocessed_dataset['label'])

    return find_best_model(X, y)


def load_dataset(src):
    """
    load images from path and write them as arrays to a dictionary, 
    together with labels and metadata.

    Parameter
    ---------
    src: str
        path to data

    Return
    ------
    dict
        Dictionary with labels, metadata and image data
    """

    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []

    for subdir in os.listdir(src):
        current_path = os.path.join(src, subdir)

        print(f'Reading files from {subdir}')
        for file in tqdm(os.listdir(current_path)):
            if file[-3:] in {'jpg', 'png'}:
                try:
                    im = imread(os.path.join(current_path, file))
                    data['label'].append(subdir)
                    data['filename'].append(file)
                    data['data'].append(im)
                except:
                    print(f'Skipping {os.path.join(current_path, file)}')

    return data


def resize_image(im, width, height):
    """
    Resize image into the specified height and width 
    using skimage.transform.resize.

    Parameter
    ---------
    im: 3d numpy array
        Image data
    width: int
        The expected output width
    height: int
        The expected output height

    Return
    ------
    3d numpy array
        The resized image data
    """
    return resize(im, (width, height))


def resize_images(images, width, height):
    """
    Resize all images into the specified width and height

    Parameter
    ---------
    images: dict
        Array with image data
    width: int
        The expected output width
    height: int
        The expected output height

    Return
    ------
    array
        The resized images data
    """
    resized_images = []
    for image in tqdm(images):
        resized_images.append(resize_image(image, width, height))

    return resized_images


class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([rgb2gray(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try:  # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


def find_best_model(X, y) -> Pipeline:
    """
    Look for the best model using GridSearchCV

    Parameter
    ---------
    X: numpy array
        Array of image data.
    y: numpy array
        Array of image labels

    Return
    ------
    Pipeline
        The model with the best score 
    """
    HOG_pipeline = Pipeline([
        ('grayify', RGB2GrayTransformer()),
        ('hogify', HogTransformer(
            pixels_per_cell=(14, 14),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys')
         ),
        ('scalify', StandardScaler()),
        ('classify', SVC())
    ])

    param_grid = [
        {
            'hogify__orientations': [7, 8, 9],
            'hogify__cells_per_block': [(2, 2), (3, 3), (4, 4)],
            'hogify__pixels_per_cell': [(8, 8), (10, 10), (12, 12)],
            'classify': [
                SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
                SVC(gamma=1e-2),
                SVC(gamma=1e-3),
                SVC(gamma=1e-4),
                SVC()
            ]
        }
    ]

    grid_search = GridSearchCV(HOG_pipeline,
                               param_grid,
                               cv=3,
                               scoring='accuracy',
                               verbose=1,
                               return_train_score=True)

    grid_res = grid_search.fit(X, y)
    return grid_res.best_estimator_
