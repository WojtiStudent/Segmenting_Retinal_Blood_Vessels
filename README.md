# Segmenting_Retinal_Blood_Vessels
**"Computer Science in Medicine" Project (Semester 5)** <br/>
**Authors:** [Wojciech Spychalski](https://github.com/WojtiStudent) [Mateusz Politycki](https://github.com/m-prezes)

This project is about different attempts to segment retinal blood vessels from special eye photos. Data to the project was taken from https://www5.cs.fau.de/research/data/fundus-images/.

## Image Processing

### About
In Image Processing attempt we decided to use only green channel of the images, because it delivers the most amount of information among chanenls and their combinations. Images were blured using Gaussian Blur, merged with original green channel. Then Frangi filter was used to detect vessels. After that image is normalized and thresholded to binary image. Finally we deleted small objects, did some combinations of errosion and dilation and filltered out all fragments that are not a part of the eye.

### Scoring
For scoring we used all metrics included in [classification_report_imbalanced](https://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.classification_report_imbalanced.html) (precision, recall, specificity, geometric mean, index balanced accuracy of the geometric mean)

### Results

### Examples


## Machine Learning


## Deep Learning
