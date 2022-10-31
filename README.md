# Segmenting_Retinal_Blood_Vessels
**"Computer Science in Medicine" Project (Semester 5)** <br/>
**Authors:** [Wojciech Spychalski](https://github.com/WojtiStudent) [Mateusz Politycki](https://github.com/m-prezes)

This project is about different attempts to segment retinal blood vessels from special photos of the eye. Data for the project were taken from https://www5.cs.fau.de/research/data/fundus-images/.

In each trail images were spilt in the same way into a train set and a test sets (last 6 pictures = test set).

## Instalation
A file named "requirements.txt" is attached to the repository. This file contains all required libraries to run the project. Run:
```
pip install requirements.txt
```
on the command line to install them all.



## Image Processing

### About
In Image Processing trial we decided to use only green channel of the images, because it provides the most information of all channels and their combinations. The images were blurred using Gaussian Blur and combined with the original green channel. Then Frangi filter was used to detect the vessels. After that iamge was normalized and thresholded to binary image. Finally we removed small objects, performed several combinations of errosion and dilation and filtered out all fragments that are not a part of the eye.

### Scoring
For scoring we used all metrics included in [classification_report_imbalanced](https://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.classification_report_imbalanced.html) (precision, recall, specificity, geometric mean, index balanced accuracy of the geometric mean)

### Results

```
Confusion matrix:

  313132  |  103150  
--------------------
  195087  | 7573975  

                   pre       rec       spe        f1       geo       iba       sup

          0       0.97      0.99      0.62      0.98      0.78      0.63   7677125
        255       0.75      0.62      0.99      0.68      0.78      0.59    508219

avg / total       0.96      0.96      0.64      0.96      0.78      0.63   8185344
```

### Example
![Image_Processing](https://user-images.githubusercontent.com/72743103/198908858-2e6d55f8-9019-4156-9576-d40ddd1d6570.jpg)

## Machine Learning

### About 
This time we decided to use Machine Learning to solve the problem. The idea was to implement something sort of rough convolutional network - we looked at a 5x5 window image and extracted some information from it (RGB mean val, RGB std, Hu moments). Then we optimized the parameters of our model (XGBoostClassifier) using [optuna](https://optuna.org). 

### Scoring
For scoring we used all metrics included in [classification_report_imbalanced](https://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.classification_report_imbalanced.html) (precision, recall, specificity, geometric mean, index balanced accuracy of the geometric mean)

### Results

```
Confusion matrix:

  97664   |  50033   
--------------------
  152617  | 1358227  

                   pre       rec       spe        f1       geo       iba       sup

          0       0.90      0.96      0.39      0.93      0.61      0.40   1408260
          1       0.66      0.39      0.96      0.49      0.61      0.35    250281

avg / total       0.86      0.88      0.48      0.86      0.61      0.39   1658541
```
### Example
![MachineLearning](https://user-images.githubusercontent.com/72743103/198909500-960971b7-af44-45f0-8671-da857ac694ec.jpg)

## Deep Learning

### About
The most complex way to segment vessels was to use Deep Learing. In this attempt we used off-the-shelf implementation of Unet ([segmentation models](https://segmentation-models.readthedocs.io/en/latest/tutorial.html)). To avoid overfitting, we used data augmentation from the [albumentations](https://albumentations.ai) library. The entire process was done using the Keras framework. This notebook was based on other notebook: https://github.com/qubvel/segmentation_models/blob/master/examples/binary%20segmentation%20(camvid).ipynb. 

### Scoring
The model was trained using sum of DiceLoss and BinaryFocalLoss, but metrics like Intersection over Union score and F1-score were also tracked.

### Results
```
Loss: 0.065215
mean iou_score: 0.96139
mean f1-score: 0.98031
```

### Example
![DeepLearning](https://user-images.githubusercontent.com/72743103/198910237-a028d1ed-c88c-4496-a297-28de6e8894be.jpg)
