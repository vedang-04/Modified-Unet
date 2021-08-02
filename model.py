#IMPORTING REQUIRED LIBRARIES.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import sys
import os
from keras.preprocessing.image import img_to_array,array_to_img
from PIL import Image
from examples_master.tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
from IPython.display import clear_output
%matplotlib inline

#Extracting the dataset and its info from Oxford_IIIT_Pet dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
