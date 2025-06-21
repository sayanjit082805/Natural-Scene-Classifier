# Natural-Scene Classifier
A deep learning image classifier built with Convolutional Neural Networks (CNN) to identify and classify natural scenes from the Intel Image Classification dataset.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)


# Overview 
This project aims to classify natural scenes/images into six categories :

- **Buildings** 
- **Forests** 
- **Glaciers**
- **Mountains**
- **Seas**
- **Streets**

# Dataset
The dataset has been taken from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). It contains around 25,000 images of natural scenes from around the world.

**Dataset Statistics** :

- **Total Images**: ~25,000
- **Training Set**: ~14,000 images
- **Test Set**: ~3,000 images
- **Validation Set**: ~7,000 images
- **Classes**: 6 (Buildings, Forest, Glacier, Mountain, Sea, Street)


# Installation

1. Clone the repository:

```bash
git clone https://github.com/sayanjit082805/Natural-Scene-Classifier.git
cd Natural-Scene-Classifier
```

2. Create a virtual environment (recommended):

```bash
python -m venv nature_classification_env
source nature_classification_env/bin/activate  # On Windows: nature_classification_env\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```


4. Download the dataset
   - Download from [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
   - Extract to `data/` directory
   - Ensure folder structure matches:
   ```
   data/
   ├── seg_train/
   │   ├── buildings/
   │   ├── forest/
   │   ├── glacier/
   │   ├── mountain/
   │   ├── sea/
   │   └── street/
   ├── seg_test/
   └── seg_pred/
   ```


For local testing, update the `train_data_path` and `test_data_path` in the notebook. Also, make sure to not run the first cell. 

I do not recommend installing locally, instead, you can just check out the [Kaggle notebook](https://www.kaggle.com/code/sayanjit082805/notebookef76f9189d).

Alternatively, you can just visit the site.

# Neural Network Architecture
The CNN consists of :

- **Input Layer** : 256x256x3 RGB images.
- **Rescaling Layer** : Rescales the input to be in the [0, 1] range. 
- **Four Convolution Layers** : 
	-  First CONV2D layer with 32 filters and a filter size of 3x3, followed by BatchNormalization and MaxPooling.
	- Second CONV2D layer with 64 filters and a filter size of 3x3, followed by BatchNormalization and MaxPooling.
	- Third CONV2D layer with 128 filters and a filter size of 3x3, followed by BatchNormalization and MaxPooling.
	- Fourth CONV2D layer with 256 filters and a filer size of 3x3, followed by BatchNormalization and MaxPooling.
- **Flattening Layer** - Flattens the output of the last convolution layer into a 1D array.
- **Dense Layer** - Consists of 128 neurons and the activation function used is the RelU activation function.
- **Output Layer** : 6 neurons, with softmax activation function for multi-class classification.

The optimiser used is the [Adam](https://keras.io/api/optimizers/adam/) optimiser, and the loss function is the sparse categorical cross-entropy. 

# Metrics
The model achieves an overall accuracy of ~83%. For more detailed metrics, please consult the classification report in the jupyter notebook.  

Due to the use of Early fitting, the model convergences at 66 epochs.

# License 
This project is licensed under The Unlicense License, see the LICENSE file for details. 

>[!NOTE]
> The License does not cover the dataset. It has been taken from Kaggle.
