# Data-Analysis-AI-and-Optimization-Project-To-bee-or-not-to-bee

[Data Analysis and AI Project Final Report.pdf](https://github.com/user-attachments/files/15593560/Data.Analysis.and.AI.Project.pdf)


# Data Analysis and AI Project
**To bee or not to bee**

Submitted on 5 June 2024

## Members of the project
- Martin JONCOURT
- Lucas SAYAG
- Chao ZHAO
- David LOISEL

## Summary
1. [Introduction](#introduction)
2. [Part 1: Feature Extraction](#part-1-feature-extraction)
    - [Symmetry Indexes of bugs](#symmetry-indexes-of-bugs)
    - [Line Ratios of bugs](#line-ratios-of-bugs)
    - [Pixel Ratios of bugs](#pixel-ratios-of-bugs)
    - [Bounding Box Proportions of bugs](#bounding-box-proportions-of-bugs)
    - [Min/Max/Mean value of red/green/blue color of bugs](#minmaxmean-value-of-redgreenblue-color-of-bugs)
    - [Median/Standard deviation of red/green/blue color of bugs](#medianstandard-deviation-of-redgreenblue-color-of-bugs)
    - [Color entropy of bugs](#color-entropy-of-bugs)
    - [Contour perimeter](#contour-perimeter)
    - [Area perimeter ratio of bugs](#area-perimeter-ratio-of-bugs)
    - [Hu moments](#hu-moments)
    - [Texture features](#texture-features)
    - [A tried feature: Flower type](#a-tried-feature-flower-type)
    - [Our full dataset](#our-full-dataset)
3. [Part 2: Data visualization](#part-2-data-visualization)
    - [Visualization](#visualization)
    - [Principal Component Analysis Projection](#principal-component-analysis-projection)
    - [Other projections](#other-projections)
4. [Part 3: Machine Learning and Deep Learning](#part-3-machine-learning-and-deep-learning)
    - [2 supervised methods that are neither deep learning nor ensemble learning](#2-supervised-methods-that-are-neither-deep-learning-nor-ensemble-learning)
    - [1 supervised ensemble learning method](#1-supervised-ensemble-learning-method)
    - [At least 2 clustering methods](#at-least-2-clustering-methods)
    - [At least 1 supervised neural network using your own features](#at-least-1-supervised-neural-network-using-your-own-features)
    - [A supervised method of your choosing trained over optimally auto-encoded features based on your extracted features](#a-supervised-method-of-your-choosing-trained-over-optimally-auto-encoded-features-based-on-your-extracted-features)
    - [Prediction phase with a batch of test images](#prediction-phase-with-a-batch-of-test-images)
    - [Bonus](#bonus)
5. [Conclusion](#conclusion)
6. [Sources](#sources)

## Introduction
Pollinator insects like bees and bumblebees are vital for ecosystems and our food supply. Our project focuses on classifying these insects using machine learning and deep learning on high-resolution images. We have a dataset of 347 insect images, with the first 250 coming with segmentation masks and a classification file. The last 97 will be used for testing. Our goal is to distinguish the various bug types.

To achieve this, we extract important features from the images, such as symmetry index, orthogonal line ratios, pixel ratios, and color statistics as well as additional features we chose. We also use visualizations like PCA and non-linear methods to understand the data better. Finally, we test various algorithms, including supervised learning, ensemble learning, clustering, and neural networks, to classify the insects.

Accurately identifying pollinator insects helps with ecological research and conservation. This project aims to create reliable methods for insect classification, aiding in monitoring pollinator populations and supporting biodiversity efforts.

## Part 1: Feature Extraction
In this first part of our project, we try to extract a number of features from our image and mask dataset. This is useful in order to give characteristics to each image and therefore, identify the bugs in each image.

### Symmetry Indexes of bugs
The symmetry index of a bug is defined by how symmetric a bug is. This implies separating the bug in 2 halves and comparing them. This index is useful as it gives information on the shape of the bug.

We first define a couple of functions:
- `load_mask`: Takes the path of a bug mask and returns an array of its pixels.
- `find_centroid`: Takes a bug mask array and returns the centroid of a bug mask.
- `rotate_image`: Takes an image array, its centroid, and an angle degree and returns a new image array rotated.
- `calculate_symmetry_index`: Takes the centroid of an image, separates the bug mask in 2 halves, flips the right half, compares its pixels to the left half and returns a value between 0 & 1 based on how different the 2 halves are.
- `find_best_symmetry`: Rotates the bug mask in several angles and for each, it calculates the symmetry index and then returns the best symmetry index among all the ones calculated.

### Line Ratios of bugs
This feature corresponds to the ratio between the longest line that fully crosses a bug and its longest orthogonal line.

We define functions:
- `find_longest_line`: Returns the length of the longest line that fully crosses the bug and its coordinates.
- `find_orthogonal_longest_line`: Returns the length and coordinates of its longest orthogonal line.
- `calculate_line_ratios`: Uses the previous functions to calculate the ratio between the 2 lines.

### Pixel Ratios of bugs
This is simply the ratio of pixels between the bug and the original image.

We define:
- `measure_pixels`: Takes an image and returns the total number of pixels.

### Bounding Box Proportions of bugs
This corresponds to the ratio between the number of pixels of a bug and the number of pixels in its bounding box.

We define:
- `compute_bounding_box_ratio`: Finds the coordinates of its vertical and horizontal bounding box lines to calculate the size of its bounding box.
- `find_optimal_rotation`: Rotates the mask in a number of angles to find the best bounding box proportion.

### Min/Max/Mean value of red/green/blue color of bugs
We determine the color range of different bugs through all these statistics of RGB channels. 

- Use OpenCV function `cv2.bitwise_and` to apply the mask to the image.
- Use numpy to calculate all statistics (Min/Max/Mean/Median/Standard deviation) of RGB channels.

### Median/Standard deviation of red/green/blue color of bugs
Similar to the above method, we use numpy for these calculations.

### Color entropy of bugs
Color entropy provides a measure of the complexity of the color distribution in an image.

- Use the formula `-np.sum(hist * np.log2(hist + 1e-9))` to calculate the entropy of the color channel.

### Contour perimeter
Perimeter is a basic shape feature that can be used to describe the outline shape of insects.

- Use OpenCV function `cv2.findContours` to find contours in the mask.

### Area perimeter ratio of bugs
This ratio can help describe the shape characteristics of bugs.

- Use `cv2.contourArea` to calculate the area of the largest contour.
- Use `cv2.arcLength` to calculate the perimeter of the largest contour.

### Hu moments
The Hu moment is a measure of the shape characteristics of an image.

- Use `cv2.HuMoments` to calculate the HuMoments.

### Texture features
Texture features are used to describe the visual patterns and structures of image surfaces.

- Contrast, Homogeneity, Energy, and Correlation are calculated using `skimage.feature graycomatrix` and `graycoprops`.

### A tried feature: Flower type
A final feature we tried to compute, but could not get through, is the flower type.

### Our full dataset
We have combined all these features into a full dataset for further analysis.

## Part 2: Data visualization
### Visualization
We use various methods to visualize the data and understand the distribution of features.

### Principal Component Analysis Projection
PCA is used to reduce the dimensionality of our dataset for better visualization.

### Other projections
We explore other non-linear projection methods to understand the data better.

## Part 3: Machine Learning and Deep Learning
### 2 supervised methods that are neither deep learning nor ensemble learning
We test two supervised learning methods on our dataset.

### 1 supervised ensemble learning method
We implement an ensemble learning method for classification.

### At least 2 clustering methods
We explore two clustering methods to group similar insects.

### At least 1 supervised neural network using your own features
We train a supervised neural network using the features we extracted.

### A supervised method of your choosing trained over optimally auto-encoded features based on your extracted features
We detail our auto-encoder architecture and training process.

### Prediction phase with a batch of test images
We use our trained models to predict the classes of a batch of test images.

### Bonus
Additional features or methods we explored during the project.

## Conclusion
Summary of our findings and the effectiveness of the methods we used.

## Sources
List of references and sources used throughout the project.
