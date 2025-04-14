## myocr

The purpose of this project is to enable us to quickly build OCR capabilities that meet our specific scenarios based on existing models or existing research. So this project provide the framework and basic capabilities.

## Installation
```
pip install .
```

## Usage
You can refer to the example code in myocr/notebooks

### Run in Local
The myocr can be incorporated as a module to your own app

### Build Custom OCR 

1. Add modeling, or trained models
2. Implement predictor for the model, you should implment the input and output for the custom model
3. Build a custom pipeline to solve specific problems

### Run as Flask Service
In main.py, we provide two rest apis, the basic OCR can be depleoyed by 
```
python main.py
```

## Build Docker Image

You can build Docker images for GPU or cpu inference.

## Acknowledge

Using the template from https://github.com/allenai/python-package-template

## License