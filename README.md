# CNN and RCNN Image Classifier

This project showcases the implementation of image classification using Convolutional Neural Networks (CNN) and VGG16 fine-tuned models. The models have been built and trained using TensorFlow and PyTorch respectively, while the web interface to predict image classes has been designed using Streamlit.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models Description](#models-description)
  - [CNN Model](#cnn-model)
  - [VGG16 Fine Tuning](#vgg16-fine-tuning)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Introduction

This project consists of two parts:
1. **CNN Model**: Implemented using TensorFlow/Keras, involving data augmentation, creating, compiling, training, and saving the model.
2. **RCNN Model using VGG16 Fine Tuning**: Implemented using PyTorch and torchvision, involving data preparation, fine-tuning, training, and saving the model.
3. **Streamlit Application**: A simple web application that allows users to upload images and predict the class using the trained CNN model.

## Requirements

- Python 3.x
- TensorFlow
- PyTorch
- Streamlit
- PIL (Pillow)
- Matplotlib
- NumPy
- Torchvision

## Installation

Clone the repository:

\```bash
git clone https://github.com/your-username/your-project-name.git
\```

Navigate to the project directory and install the required packages:

\```bash
cd your-project-name
pip install -r requirements.txt
\```

## Usage

### Run the Streamlit Application
Start the Streamlit application by running:

\```bash
streamlit run app.py
\```

Access the application via a web browser at:

\```plaintext
http://localhost:8501
\```

### Run the Jupyter Notebook
Open the Jupyter Notebook to train and test the models:

\```bash
jupyter notebook your-notebook-name.ipynb
\```

## Models Description

### CNN Model
The CNN model is implemented using TensorFlow/Keras with the following architecture:

- Convolution layers with ReLU activation functions.
- Max-pooling layers for downsampling.
- Fully connected Dense layers.
- Softmax activation in the output layer.

### VGG16 Fine Tuning
The VGG16 model has been fine-tuned to suit the specific classification task. This involves:

- Loading a pre-trained VGG16 model.
- Modifying the classifier to match the number of classes.
- Fine-tuning using the SGD optimizer and Cross-Entropy Loss.

## References

1. [TensorFlow Documentation](https://www.tensorflow.org/)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
3. [VGG16 - Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
4. [Streamlit - The fastest way to build custom ML tools](https://www.streamlit.io/)
5. [Keras - The Python Deep Learning API](https://keras.io/)
6. [Pillow - Python Imaging Library (Fork)](https://pillow.readthedocs.io/en/stable/)
7. [Matplotlib - A plotting library for Python](https://matplotlib.org/)
8. [NumPy - The fundamental package for scientific computing with Python](https://numpy.org/)
9. [Torchvision - Datasets, Transforms and Models specific to Computer Vision](https://pytorch.org/vision/stable/index.html)
10. [OpenAI ChatGPT - A State-of-the-Art Language Model](https://openai.com/research/chatgpt)
