# Deep Learning Object Detection
Developed a deep learning predictive model that can determine, given an intersection image, the class and location of the objects objects of two types (car or truck).

## Introduction

When humans look at an image, they're able to analyze the image, process the contents and understand what they're seeing and are instantly able to connect the objects in the image to objects they know. But how does a computer do this? To a computer, an image is a collection of RGB pixels; so, how exactly does a computer look at an image and decipher the objects within the image? This is one of the fundamental problems of computer vision known as object detection where an image is input, and the output would be a precise estimation of the classification of the objects as well as the location of the object classified in the image which are usually represented as bounding boxes. There has been considerable amount of work done for object detection using deep learning techniques. Neural networks adept to object detection have been trained with high accuracy rates on large datasets. 

## Aim

In this project, the main objective was to learn to partially re-train a convolutional neural network adept at object detection to only detect two kinds of objects: cars and trucks. The input to our trained model would be an image and the model would predict the classification as well as locations of the cars and/ or trucks within the image. 

<img width="958" alt="Screen Shot 2021-06-22 at 3 23 03 PM" src="https://user-images.githubusercontent.com/32781544/122916293-943f1900-d311-11eb-850e-a1197da93c7e.png">

## Model

For this project, transfer learning was used to retrain an object detection model to only detect two classes: cars and trucks. The model I used was RetinaNet which is a one-stage detector and makes use of focal loss. The backbone for RetinaNet that was used was ResNet50 which does feature extraction. In addition to this, there are two subnetworks for classification and bounding box regression; this forms the RetinaNet. 

<p class="aligncenter">
  <img width="770" alt="Screen Shot 2021-06-22 at 3 21 43 PM" src="https://user-images.githubusercontent.com/32781544/122916131-678b0180-d311-11eb-9839-5a11b07127e1.png">
</p>

When analysing the data before beginning the training, I noticed a major difference in the number of instances of cars and trucks in both the training and validation datasets. Since the RetinaNet loss function deals with class imbalance, I felt it was the best model to use for this project.

The RetinaNet implementation called "keras-retinanet" would have to be installed as well as pre-trained weights for this network. Using the framework and weights, you would re-train the model using custom training and validation annotation text files using the training script. The evaluation script can be used to test the model and predictions. Instructions to run the file can be found in "ReadMe.txt" uploaded with the python scripts.
