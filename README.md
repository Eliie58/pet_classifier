# Pet Classifier

Image processing project for favorite Pet classification.

## Environment

To create and configure the conda environment, please foloow these steps:

- Clone the repository to your local machine.
- From the root of the project, run the following:
  ```
  conda create --name pet_classifier
  conda activate pet_classifier
  python3 -m pip install -r requirements.txt
  ```

## Build Dataset

To compile the datasets, follow the steps in the [notebook](notebooks/Build_Dataset.ipynb).
TO launch the notebook, run the following command from the root of the directory:

```
python3 -m jupyterlab
```

## Notebooks

Under notebooks, there are 2 notebooks:

- Build_Dataset: Used to collect images and build the dataset
- Pet_classifier: Used to understand the process of building the model, and improving the performance.

## Scripts

Under scripts, there are the functions to train the model, and predict the results.
To use the functions, from python shell in the scripts folder:

```
from train import build_model
from predict import predict

model = build_model(path_to_your_dataset)
predict(model, path_to_your_target_image)
```
