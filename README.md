# Mathematical-Symbol-Recognition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

Explanation:
This project utilizes a Convolutional Neural Network (CNN) for classifying grayscale images of handwritten math symbols into one of 82 possible classes. The steps taken include:

Data Preprocessing:

Reading images from the dataset.
Resizing and normalizing them.
Label encoding for categorical output.

Model Building:

A CNN architecture using layers like Conv2D, MaxPooling, Dropout, and Dense.
ReLU activation for hidden layers and Softmax for multi-class output.
Model Training and Evaluation:
Trained over 10 epochs using the Adam optimizer.
Evaluation done using accuracy and loss metrics.
Confusion matrix plotted to visualize performance across classes.

Results:
Achieved a reasonable accuracy considering the complexity and number of classes.
Observed common misclassifications between visually similar symbols.

DataSet Link : The dataset used in this project is ‘Kaggle - Handwritten Math Symbols’, available at: 📎 https://www.kaggle.com/xainano/handwrittenmathsymbols
