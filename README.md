##Data science Internship based on 2018 Data Science Bowl
####Dataset:
[2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018/data)


####Goal:
Build a semantic segmentation model with UNet architecture using Keras.

####Main techniques:
* U-net architecture of model was inspired by [this paper](https://arxiv.org/abs/1505.04597)
* Data augmentation
* Model checkpoint(saving best model)
* Custom Dice coefficient metric
* Custom loss function, which is a combination of binary cross entropy and dice coefficient 

####Data augmentation
Data augmentation in this project was done using next techniques: rotation, width/height shift, shear, zoom, horizontal/vertical flip.
####Metrics
As a metric for this project a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) was used.
In the result, an average Dice coefficient of 0.88 was achieved.
####Loss function
Combination of binary cross entropy and dice coefficient metric showed itself as a better choice of loss function, rather than pure binary cross entropy.
####Installation
Run ```pip install -r requirements.txt```  to install all dependencies.
####Training
Run ```python train.py``` to train model.
####Predictions
Run ```python predict_masks.py``` to predict masks. All the predictions will be stored in directory named output.
####Model evaluation
Run ```python model_evaluation.py``` to evaluate model on validation set.