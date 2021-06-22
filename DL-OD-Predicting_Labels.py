#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import csv
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

THRESH_SCORE = 0.6

import argparse

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", help="Path to saved model")
parser.add_argument("--rootpath", help="Path to save prediction file")
parser.add_argument("--testpath", help="Path to test set directory")

# Read arguments from command line
args = parser.parse_args()


#Change to path of test set
root_path = args.rootpath
test_path = args.testpath
model_path = args.modelpath

print(model_path)

model = models.load_model(model_path, backbone_name='resnet50')
model = models.convert_model(model)

def predict(some_image):
    some_image = preprocess_image(some_image.copy())
    some_image, scale = resize_image(some_image)

    some_boxes, some_scores, some_labels = model.predict_on_batch(
    np.expand_dims(some_image, axis=0)
    )

    some_boxes /= scale

    return some_boxes, some_scores, some_labels


def unnorm_data(xmin, ymin, xmax, ymax):
    w = xmax - xmin
    h = ymax - ymin

    cx = xmin + (w / 2)
    cy = ymin + (h / 2)

    width, height = 640, 640

    cx /= width
    w /= width

    cy /= height
    h /= height

    return cx, cy, w, h


with open(root_path + 'predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=' ')

    for i in range(1, 1001):
        print("Image", i)
        image = read_image_bgr( test_path + str(i) + '.jpg')
        boxes, scores, labels = predict(model1, image)

        for box, score, label in zip(boxes[0], scores[0], labels[0]):

            cx, cy, w, h = unnorm_data(box[0], box[1], box[2], box[3])
            writer.writerow([i, label, cx, cy, w, h, score])


