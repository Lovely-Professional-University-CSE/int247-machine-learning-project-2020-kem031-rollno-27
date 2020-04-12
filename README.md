# int247-machine-learning-project-2020-kem031-rollno-27
int247-machine-learning-project-2020-kem031-rollno-27 created by GitHub Classroom

## Project Title:  "Handwritten digit recognition"
#### Project description: "MNIST datasets by Keras and datasets downloaded from Kaggle have been used in this project. Motive of this project is to find out the model that can perform with best accuracy."
## Team Members: 1 
#### Team description: Sriashika Addala, Roll no 27, KEM031
## Website URL
<a href="https://sites.google.com/view/digitrecognition-kem031-27/home"> Click to view project site </a>

Samples provided from MNIST (Modified National Institute of Standards and Technology) dataset includes handwritten digits total of 70,000 images consisting of 60,000 examples in training set and 10,000 examples in testing set, both with labeled images from 10 digits (0 to 9). This is a small segment form the wide set from NIST where size was normalized to fit a 20*20 pixel box and not altering the aspect ratio. Handwritten digits are images in the form of 28*28 gray scale intensities of images representing an image along with the first column to be a label (0 to 9) for every image. The same has opted for the case of the testing set as 10,000 images with a label of 0 to 9.

## CNN Model
Excellent results achieve a prediction error of less than 30%. State-of-the-art prediction error of approximately 0.2% can be achieved with large Convolutional Neural Networks. <b> Input & Output: </b>
When a computer or system takes an image, it just sees an array of pixel values. Suppose 480 * 480 * 3 where (480*480) is size, 3 refers to RGB values. Each of these numbers is assigned with a value of 0 to 255 as pixel intensities at that point. The key point is that based on taking the image as an input, computer system predicts and make an assumption as output for describing the probability of the image being a said or certain class (say 0.90 for class 1, 0.96 for class 2, 0.4 for class 3).
<p> Using CNN model, we have achieved a 98% training accuracy and 93% testing accuracy with a maximum testing loss of 27%. </p>

## KNN model
KNN is the non-parametric method or classifier used for classification as well as regression problems. KNN explains categorical value using majority votes of K nearest neighbors where the value for K can differ, so on changing the value of K, the value of votes can also vary. Different distance functions used in KNN are-
  1.  Euclidean function
  2.  Manhattan function
  3.  Minkowski
  4.  Hamming distance
  5.  Mahalanobis distance
<p> Using KNN classifier, we have achieved a 86% training accuracy and 77% testing accuracy. </p>

## Random Forest Classifier
Random forests are an ensemble learning method that can be used for classification. It works by using a multitude of decision trees and it selects the class that is the most often predicted by the trees. Each tree of the forest is created using a random sample of the original training set, and by considering only a subset of the features (typically the square root of the number of features). The number of trees is controlled by cross-validation. Using RandomForestClassifier, we have achieved 95.7% training accuracy and 87.3% testing accuracy.

## NN model
We have used the scikit learn's in-built digits dataset and have achieved a 92% training and testing accuracy.
