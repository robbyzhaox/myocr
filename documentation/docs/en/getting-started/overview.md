## Overview
========


## MyOCR Components

![MyOCR Componants](../assets/images/components.png)


**Model** is artificial neural network defined by Pytorch, it consists two parts, the model architecture and model weights, a model architecture usually build with transform, backbone, neck and head. MyOCR also have a tool for easily train the model by custom data.

The Model Class has serval sub classes: PytorchModel, OnnxMode, CustomModel.

**Converter** is a component for building Predictor, a Converter is responsible for preparing appropriate parameters for a model, and then convert the model output to a specific type.

**Predictor** is built on Model and Converter, it is a component to do a real world task by the neural network. For example, we have a MLP model for classifing images, usually the model accepts a tensor represents the image and output a vector representing the probability of each class, but for a user we just want give an image and got the class name with a confidence, thus a converter will help to eliminate the gap of the input and output we want and network provides. By using a Model and a Predictor we will got a Predictor for us to do a real world task, for the above image classification task, the input for Predictor can be an image and output can be a class name.

**Pipeline** is the arrangement and combination of different Predictors. It is used to solve a more complex problem, which may have multiple steps and require a combination of multiple models to complete. Such as for a traditional image OCR task, we at least need a detection model and a recognition model, detection model to find where there is text in the image, and recognition model to extract specific text content.
