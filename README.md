## Engineering School ISEP Project
# Data Analysis and AI Project
**To bee or not to bee**

[Data Analysis and AI Project Final Report.pdf](https://github.com/user-attachments/files/15593560/Data.Analysis.and.AI.Project.pdf)

Submitted on 5 June 2024

Mark: 19/20

## Members of the project
- Martin JMP
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
3. [Part 2: Data Visualization](#part-2-data-visualization)
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

## Part 2: Data Visualization
### Visualization
To better understand the distribution and relationships within our data, we employed various visualization techniques. These visualizations help identify patterns, anomalies, and insights that guide our further analysis and model building.

- **Histograms**: We plotted histograms for each feature to observe their distributions. This helped us understand the spread and central tendency of the data.
- **Pairwise Scatter Plots**: Scatter plots between pairs of features helped us see potential correlations and clusters in the data.
- **Heatmaps**: Correlation heatmaps were used to identify relationships between features. High correlation values indicate features that are potentially redundant or provide similar information.

### Principal Component Analysis Projection
Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms our high-dimensional data into a lower-dimensional space while preserving as much variance as possible. This helps in visualizing the data in 2D or 3D plots, making it easier to identify clusters and patterns.

- We applied PCA to our dataset and plotted the first two principal components.
- This visualization showed how well-separated the different classes of insects are based on the extracted features.
- PCA also helped in identifying the most significant features contributing to the variance in the data.

### Other Projections
Apart from PCA, we explored other non-linear dimensionality reduction methods to capture more complex relationships in the data.

- **t-SNE (t-distributed Stochastic Neighbor Embedding)**: t-SNE is particularly good at preserving local structure in the data and can reveal clusters that PCA might miss. We applied t-SNE to our dataset and visualized the results.
- **UMAP (Uniform Manifold Approximation and Projection)**: UMAP is another powerful technique for non-linear dimensionality reduction. It tends to produce more meaningful global structures compared to t-SNE. We used UMAP to visualize our data and compared it with the PCA and t-SNE results.

## Part 3: Machine Learning and Deep Learning
### 2 Supervised Methods that are Neither Deep Learning nor Ensemble Learning
We implemented and evaluated two traditional supervised learning methods on our dataset.

- **Support Vector Machines (SVM)**: SVMs are effective for high-dimensional spaces and are commonly used for classification tasks. We trained an SVM classifier on our features and evaluated its performance.
- **K-Nearest Neighbors (KNN)**: KNN is a simple, instance-based learning method. We used KNN to classify the insects based on their extracted features and evaluated its accuracy.

### 1 Supervised Ensemble Learning Method
Ensemble learning methods combine multiple models to improve performance. We implemented the following ensemble method:

- **Random Forest**: Random Forest is an ensemble method that uses multiple decision trees to improve classification accuracy and control overfitting. We trained a Random Forest classifier and assessed its performance on the dataset.

### At Least 2 Clustering Methods
Clustering helps in grouping similar data points without prior knowledge of the labels. We explored the following clustering methods:

- **K-Means Clustering**: We applied K-Means clustering to our dataset to identify natural groupings of insects based on the extracted features. The optimal number of clusters was determined using the Elbow method.
- **Agglomerative Hierarchical Clustering**: This method builds a hierarchy of clusters by recursively merging or splitting them. We used dendrograms to visualize the clustering process and determine the optimal number of clusters.

### At Least 1 Supervised Neural Network Using Your Own Features
Neural networks can capture complex relationships in the data. We trained the following neural network model:

- **Feedforward Neural Network**: Using the extracted features as inputs, we designed and trained a feedforward neural network. The architecture included input layers corresponding to our features, several hidden layers with ReLU activations, and an output layer with softmax activation for classification. The model was trained using backpropagation and evaluated on a validation set.

### A Supervised Method of Your Choosing Trained Over Optimally Auto-encoded Features Based on Your Extracted Features
Autoencoders are used for unsupervised learning of efficient codings. We trained an autoencoder and used the encoded features for classification.

- **Autoencoder**: We designed an autoencoder with an encoder part that compresses the feature space into a lower-dimensional representation and a decoder part that reconstructs the input from the encoded representation.
- **Classification**: The encoded features from the autoencoder were then used to train a supervised classifier. We experimented with different classifiers such as logistic regression, SVM, and neural networks to determine the best-performing model.

### Prediction Phase with a Batch of Test Images
After training our models, we evaluated their performance on a separate batch of test images to simulate real-world usage.

- **Test Images**: The test set consisted of 97 images that were not part of the training set.
- **Prediction**: Each model was used to predict the class of insects in the test images. The predictions were compared with the ground truth labels to evaluate accuracy, precision, recall, and F1-score.
- **Ensemble Prediction**: We also combined predictions from multiple models using voting techniques to improve overall performance.

### Bonus
During the project, we explored additional features and methods that provided further insights or improvements.

- **Ensemble of Autoencoders**: We experimented with combining multiple autoencoders to capture different aspects of the data.
- **Advanced Feature Engineering**: We explored additional texture features, shape descriptors, and color histograms that were not part of the initial feature set.

## Conclusion
Through this project, we successfully developed a comprehensive pipeline for classifying pollinator insects using both traditional and advanced machine learning techniques. By extracting meaningful features from high-resolution images and leveraging various visualization, clustering, and classification methods, we achieved a high level of accuracy in distinguishing different types of insects.

Our findings demonstrate the effectiveness of combining domain-specific feature extraction with powerful machine learning algorithms. The use of PCA, t-SNE, and UMAP provided valuable insights into the data structure, while supervised learning methods, including neural networks and ensemble methods, delivered robust classification performance.

Future work could explore real-time insect monitoring systems, integrating these models into mobile applications or IoT devices for ecological research and conservation efforts. Our project contributes to the ongoing efforts in preserving pollinator populations and supports biodiversity through accurate and automated insect classification.

## Sources
- [OpenCV documentation](https://opencv.org/)
- [NumPy documentation](https://numpy.org/)
- [scikit-learn documentation](https://scikit-learn.org/)
- [PCA and t-SNE](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Random Forest](https://link.springer.com/article/10.1023/A:1010933404324)
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Autoencoders](https://www.deeplearningbook.org/)
- [TensorFlow documentation](https://www.tensorflow.org/)
- [PyTorch documentation](https://pytorch.org/)
