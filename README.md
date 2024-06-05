# Data-Analysis-AI-and-Optimization-Project-To-bee-or-not-to-bee

[Data Analysis and AI Project.pdf](https://github.com/user-attachments/files/15593560/Data.Analysis.and.AI.Project.pdf)




Data Analysis and AI Project
To bee or not to bee
Submitted on 5 June 2024
Members of project
Martin JONCOURT
Lucas SAYAG
Chao ZHAO
David LOISEL
1
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Summary
Introduction............................................................................................................................ 3
Part 1 Feature Extraction.......................................................................................................4
1) Symmetry Indexes of bugs.............................................................................................5
2) Line Ratios of bugs........................................................................................................ 7
3) Pixel Ratios of bugs....................................................................................................... 9
4) Bounding Box Proportions of bugs...............................................................................11
5) Min/Max/Mean value of red/green/ blue color of bugs................................................. 13
6) Median/Standard deviation of red/green/ blue color of bugs........................................13
7) Color entropy of bugs...................................................................................................14
8) Contour perimeter........................................................................................................ 15
9) Area perimeter ratio of bugs.........................................................................................16
10) Hu moments...............................................................................................................17
11) Texture features..........................................................................................................18
12) A tried feature : Flower type.......................................................................................19
13) Our full dataset...........................................................................................................22
Part 2 Data visualization......................................................................................................24
1) Visualization................................................................................................................. 24
2) Principal Component Analysis Projection.................................................................... 27
3) Other projections..........................................................................................................29
Part 3 Machine Learning and Deep Learning.................................................................... 32
● 2 supervised methods that are neither deep learning nor ensemble learning:............. 33
● 1 supervised ensemble learning method [II.2413]........................................................ 35
● At least 2 clustering methods [II.2413]..........................................................................36
● At least 1 supervised neural network using your own features [IG.2411] :................... 38
● A supervised method of your choosing trained over optimally auto-encoded features
based on your extracted features. You will detail your auto-encoder architecture and
training process [IG.2411] :.............................................................................................. 41
● Prediction phase with a batch of test images:...............................................................49
● Bonus :..........................................................................................................................53
Conclusion............................................................................................................................56
Sources................................................................................................................................. 57
2
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Introduction
Pollinator insects like bees and bumblebees are vital for ecosystems and our food
supply. Our project focuses on classifying these insects using machine learning and
deep learning on high-resolution images. We have a dataset of 347 insect images,
with the first 250 coming with segmentation masks and a classification file. The last
97 will be used for testing. Our goal is to distinguish the various bug types
To achieve this, we extract important features from the images, such as symmetry
index, orthogonal line ratios, pixel ratios, and color statistics as well as additional
features we chose. We also use visualizations like PCA and non-linear methods to
understand the data better. Finally, we test various algorithms, including supervised
learning, ensemble learning, clustering, and neural networks, to classify the insects.
Accurately identifying pollinator insects helps with ecological research and
conservation. This project aims to create reliable methods for insect classification,
aiding in monitoring pollinator populations and supporting biodiversity efforts.
3
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Part 1 Feature Extraction
In this first part of our project, we try to extract a number of features from our image
and mask dataset. This is useful in order to give characteristics to each image and
therefore, identify the bugs in each image.
Before extracting, we data pre-process. There is a mask missing, so we delete its
corresponding image.
Fig 1 : Image number 154
We tried creating a mask of this image ourselves but it proved fruitless as the mask
created identified the flower on the top page as the target. This difficulty is likely due
to the presence of 2 different bugs on the image. Regardless, we moved forward
without this image.
4
Data Analysis and AI Project 06/05/2024
To bee or not to bee
1) Symmetry Indexes of bugs
The symmetry index of a bug is defined by how symmetric a bug is. This implies
separating the bug in 2 halves and comparing them. This index is useful as it gives
information on the shape of the bug. Here is how we calculate them :
We first define a couple of functions :
- load_mask : This function takes the path of a bug mask and returns an array
of its pixels. It is useful in order to load a bug mask.
- find_centroid : This one takes a bug mask array and returns the centroid of a
bug mask.
- rotate_image : This function takes an image array, its centroid and an angle
degree and returns a new image array rotated. It is necessary in order to
rotate the bug mask in several different axes.
- calculate_symmetry_index : This function takes the centroid of an image,
separates the bug mask in 2 halves, flips the right half, compares its pixels to
the left half and returns a value between 0 & 1 based on how different the 2
halves are. The result is then deduced from 1 giving a symmetry index where
if close to 0, the symmetry is low, and if close to 1, the symmetry is high.
- find_best_symmetry : This function rotates the bug mask in several angles
and for each, it calculates the symmetry index and then returns the best
symmetry index among all the ones calculated.
Then, we combine these functions in order to load every bug mask, find the centroid
for each, find the best symmetry for each and return a dataset df of all the symmetry
indexes with the corresponding bug mask indexes. This is df :
Fig 2 : Symmetry index
We also use the describe function in order to have a better idea of this dataset :
5
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 3 : Symmetry index description
We also write a code that returns the images with the highest and lowest symmetry
indexes. This is useful in order to make sure of the validity of the code.
Fig 4 : Highest and lowest symmetry indexes
As we can see, the values range between 0 and 0.89. These low symmetry indexes
indicate that most bugs are not symmetric. We now have a feature for all the images.
We notice that this code takes a good 4 hours to run. This is because it requires to
work on all 249 images and for each, calculate the symmetry index on 37 different
angles.
6
Data Analysis and AI Project 06/05/2024
To bee or not to bee
2) Line Ratios of bugs
Now, let’s compute another feature : Line ratio. This feature corresponds to the ratio
between the longest line that fully crosses a bug and its longest orthogonal line (the
smallest of the 2, divided by the longest of the 2). This is useful to get a better idea of
the shape of a bug. To do so, here is how we calculate it :
We first define a coupe of functions :
- find_longest_line : This function takes a bug mask and returns the length of
the longest line that fully crosses the bug as well as its coordinates. This is
necessary to find the longest line.
- find_orthogonal_longest_line : This function takes a bug mask as well as its
results of the find_longest_line function and returns the length and
coordinates of its longest orthogonal line.
- calculate_line_ratios : This function takes a bug mask, uses the previous 2
functions and calculates the ratio between the 2
Then, we apply the calculate_line_ratios functions to all the masks and add a line
ratio column to df :
Fig 5 : Line ratio
7
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Once again, we describe that dataset :
Fig 6 : Line ratio description
We also write a code that once again returns the images with the highest and lowest
line ratios, useful to make sure of the validity of our code.
Fig 7 : Highest and lowest line ratios
As we can see, the values range between 0 and 1 (since this is a list of ratios) and
the values are quite varied and high (the mean and median are around 0.8). This
indicates that most bugs are not long but have a rounder shape. However, the
variation of data is useful as it makes this feature discriminating and helps separate
bugs.
Once again, this code takes a long time to run, a good 5 hours. The reasons behind
this long run are the same as for the first feature : it takes 249 images and rotates
each image 37 times.
8
Data Analysis and AI Project 06/05/2024
To bee or not to bee
3) Pixel Ratios of bugs
Another ratio feature we can calculate is the pixel ratio. This is simply the ratio of
pixels between the bug and the original image. This feature is quite useful as it helps
separate large bugs from the smaller ones. Here is how we calculate it :
First, we defined a simple function :
- measure_pixels : takes an image and returns the total number of pixels
Then, we calculate for each image and corresponding mask the number of pixels,
divide the number of pixels in the mask by the number of pixels in the image and
once again we create a dataframe with all the results which we merge to the original
dataframe df.
Fig 8: Pixel ratio
9
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Additionally, we describe the dataset :
Fig 9 : Pixel ratio description
Finally, we look for the bug masks with the highest pixel ratio and the lowest and
return them :
Fig 10 : Highest and lowest pixel ratio
This time, the values only range between 0.003 and 0.18. This may seem small but it
is coherent and makes sense that even the largest bug doesn’t take 20% of an
image (which still remains large for a bug proportion). However, the variation in
values still greatly helps us distinguish bugs from one another and adds a new
feature to our dataset.
This time, the code is quite fast and only takes a couple of minutes to run. This is
because it is simpler and only requires calculating the number of pixels for 498
images.
10
Data Analysis and AI Project 06/05/2024
To bee or not to bee
4) Bounding Box Proportions of bugs
Still in the themes of shapes, we decide to add another feature which is the
bounding box proportions of bugs. This corresponds to the ratio between the
number of pixels of a bug and the number of pixels in its bounding box (defined by
the 2 most distant points of a bug). This feature is quite useful as it gives an idea on
the shape of the bug and differentiates bugs with spikes or limbs from rounder bugs.
In order to compute it, we did the following :
First we define a couple of functions :
- compute_bounding_box_ratio : this function takes a bug mask, finds the
coordinates of its vertical bounding box lines and of its horizontal bounding
box lines in order to calculate the size of its bounding box. It then calculates
the size of its bug mask and returns the ratio between its bug mask size and
bounding box size (in other words, the bounding box proportion of the bug).
- find_optimal_rotation : this function is used to rotate the mask in a number
of angles using the rotate_image function in order to find the best bounding
box proportion. This is necessary as the initial compute_bounding_box_ratio
function is limited to the original rotation of the bug and would not work with
the ‘proper’ bounding box.
Then, for each bug, we calculate the bounding box proportion and once again create
a list with all the values and added it to the dataframe df :
Fig 11 : Bounding Box Proportions
11
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Once again, we describe that dataset :
Fig 12 : Bounding Box Proportion description
Finally, we look for the bug masks with the highest and lowest bounding box
proportions in order to validate our code :
Fig 13 : Highest and lowest bounding box proportions
As we can see, the values are quite varied and range between 0.19 and 0.75 giving
us a good image separator feature.
This code once again takes a long time to run. This is because it uses the
rotate_image function which implies going through 37 different degrees in order to
find the optimal bounding box.
12
Data Analysis and AI Project 06/05/2024
To bee or not to bee
5) Min/Max/Mean value of red/green/ blue color of bugs
6) Median/Standard deviation of red/green/ blue color of
bugs
We determine the color range of different bugs through all these statistics of RGB
channels. It not only can help the classifier distinguish between different types of
bugs，but also can help us more accurately separate the bee from the background in
the image segmentation task.
- Use OpenCV function cv2.bitwise_and to apply the mask to the image,
resulting in an image masked_image containing only the area covered by the
mask
- Use numpy to calculate all statistics (Min/Max/Mean/Median/Standard
deviation) of RGB channels.
Fig 14 : Color statistics 1
13
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 15 : Color statistics 2
Color Statistics can only provide a general overview of the image colors and cannot
reflect the details and local information in the image. For example, the mean may
mask important local features in the image.For images with complex textures or rich
details, color statistics alone may not be enough to capture their characteristics.
7) Color entropy of bugs
Color entropy provides a measure of the complexity of the color distribution in an
image. A high entropy value indicates a complex color distribution with more color
variations. A low entropy value indicates a simple color distribution with more uniform
colors. These features can be used to train machine learning models to help
distinguish different types of bugs.
- Use the formula -np.sum(hist * np.log2(hist + 1e-9)) to calculate the entropy of
the color channel. 1e-9 is a small constant to avoid taking the logarithm of
zero.
14
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 16 : Color entropy
The color entropy of most images is between 6.5 and 7.5, which is a relatively high
entropy value (usually entropy = 0 means pure color), which means that the image is
relatively complex and contains a lot of information and details. At the same time, the
distribution of color entropy is relatively concentrated, which may not be a good
feature.
8) Contour perimeter
Perimeter is a basic shape feature that can be used to describe the outline shape of
insects. Different species of insects may have different perimeter features.
- Use OpenCV function cv2.findContours to find contours in the mask,
cv2.RETR_EXTERNAL retrieves only the external contours, and
cv2.CHAIN_APPROX_SIMPLE removes redundant points in the contours.
15
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 17 : Image perimeter
Shape complexity: The contour perimeter can only reflect the length of the object's
boundary and cannot provide other important information about the object's shape,
such as area, shape compactness, etc.For this dataset, this feature is greatly
affected by the bug posture and may not be a good feature.
9) Area perimeter ratio of bugs
This ratio can help describe the shape characteristics of bugs. For different species
of insects, their shape characteristics may vary significantly.
- Use cv2.contourArea to calculate the area of the largest contour,
cv2.arcLength to calculate the perimeter of the largest contour
16
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 18 : Area-perimeter ratio
In general, compact shapes (like circles) will have a higher area-to-perimeter ratio,
while complex or irregular shapes will have a lower area-to-perimeter ratio.Usually
this is a good feature, but in this dataset, this feature is still greatly affected by the
insect's posture. For example, pictures 1 and 3 are of the same bug, but due to
different postures, the area-to-perimeter ratio is quite different.
Picture 1 Area-perimeter ratio =24.04 Picture3 Area-perimeter ratio =44.95
10) Hu moments
The Hu moment is a measure of the shape characteristics of an image. It is rotation,
scaling, and mirror invariant, which means that even if the insect appears in different
postures in the image, the Hu moment can still effectively describe its shape
characteristics.
- Use cv2.HuMoments to calculate the huMoments.
17
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 19 : Hu moments
The values of each Hu moment vary greatly in different images. For example, Hu
Moment 5 and Hu Moment 7 are negative in some images and positive in others,
which reflects the diversity of shape features.This might be a good feature.
11) Texture features
Texture features are an important concept in image processing and computer
vision, which are used to describe the visual patterns and structures of image
surfaces.
- Contrast measures the local intensity variation between gray levels. High
contrast indicates that there are significant intensity variations in the image,
reflecting the complexity of the image.
- Homogeneity measures the local similarity of elements in the gray-level
co-occurrence matrix (GLCM). High homogeneity means that neighboring
pixel values are similar, reflecting the smoothness of the image.
- Energy measures the uniformity of the element values in the gray-level
co-occurrence matrix (GLCM). High energy indicates the presence of
repeated gray-level patterns in the image.
- Correlation measures the statistical relatedness of pairs of pixels in the Gray
Level Co-occurrence Matrix (GLCM). High correlation indicates linear
dependence of neighboring pixel values.
- Use skimage.feature graycomatrix and graycoprops functions to calculate
gray-level co-occurrence matrix and texture features
18
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 20 : Textures
From this result, we can see that there is no big difference in Homogeneity,
Energy and Correlation between different images. This means that these 3
features may not be a significant feature for this dataset
12) A tried feature : Flower type
A final feature we try to compute, but can not get through, is the flower type. This
feature would have been useful as there are 31 different flower types among the
images and bugs don’t pollinate the same flowers. In order to achieve this objection,
we first identify all the flowers in every image using a website called Pl@ntNet. Then
for each type of flower, we add a list of characteristics which is useful in order to
identify the bugs. This is the list :
19
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 21 : Flower types and characteristics
We now have knowledge on the flower types. We then try to create a code that
identifies the flower type in each image. To this end, we first the masks and apply
them to each image in order to have images of only the background of images,
without the bugs :
20
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 22 : 1 reversed mask 64
Fig 23 : 1 reversed image 64
However, the problem rises when we try to identify the flower types in each image.
Indeed, trying to use the characteristics of each flower type proves fruitless. The
problem is that it is difficult to distinguish the flower from the rest of the image. We try
other ways such as using the pixels touching the bugs (as the bugs are on the
flowers in most photos) but that does not work either (and not all bugs touch their
flowers). Finally, we notice that in most photos, the flowers are less blurry than their
background. We try to work with that but to no end since the odd shapes of flowers
make their blurriness difficult to call.
In the end, we are letting go of that feature, despite its usefulness. We could add a
feature column of all flower types identified manually in each image but we believe
that all our features should have to be computed in Python directly.
21
Data Analysis and AI Project 06/05/2024
To bee or not to bee
13) Our full dataset
Regardless of that, we now have a number of features to identify the bugs in each
image.
22
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 24 : Our full dataset of features
What remains now is what we are going to do with them.
23
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Part 2 Data visualization
Having these features in our hands, we can now use them to create visuals in order
to have a better idea of their importance.
1) Visualization
However, we first need to visualize the bug types and species. Indeed, we have a list
of the bug types and species of each image. Using this information, we first describe
the dataset with bug types and species :
Fig 25 : Description of bug types and species
As we can see, the most prominent bug type is the bee and the most prominent
species is Bombus hortorum (which is not a bee).
However, more importantly, we create repartition visuals including a pie plot of all the
bug types in order to see which are the most prominent :
Fig 26 : Distribution of bug types
24
Data Analysis and AI Project 06/05/2024
To bee or not to bee
As we can see, the most prominent bugs are by far and away bees and bumblebees.
Obviously, the main goal of our project is to identify these 2 bugs so it makes sense
that our dataset is mostly populated with them. However, there are still 5 other bug
types in this dataset
Additionally, we decide to go into further detail and look at the repartition of species.
To this end, we create 2 bar plots incorporating both bug types and species :
25
Data Analysis and AI Project 06/05/2024
To bee or not to bee
We first notice that every species belongs to 1 bug type. Furthermore, both bees and
bumblebees are in large numbers but not in the same way. Indeed, while bees have
more species than bumblebees, it’s Bombus hortorum (a bumblebee species) which
is in highest number.
From these visuals, we can expect that bees and bumblebees will be easier to
distinguish from other species as having a lot of them will help define them and help
our algorithms identify them once we receive new images.
26
Data Analysis and AI Project 06/05/2024
To bee or not to bee
2) Principal Component Analysis Projection
Having visuals for the bug types and species is useful. However, we now decide to
make visuals using the features gathered during part 1. To do so, we first
standardize our features using StandardScaler. We then find the 2 highest explained
variance which became our principal components. Finally, we perform a Principal
Component Analysis which we plot :
As we can see, bug types are somewhat difficult to distinguish but Bumblebees
generally have a higher line ratio than bees. Additionally, despite being in few
numbers, the dragonfly stands out by having by far the highest line ratio. The
percentage of variance does not amount to 100% but this is because we are working
with 33 features.
27
Data Analysis and AI Project 06/05/2024
To bee or not to bee
In addition to this PCA, we plot a correlation circle :
This correlation circle helps us noice the correlations between variables. Indeed,
while line ratio and symmetry index are uncorrelated but the Hu Moments are very
correlated to the line ratio.
Variables like "Symmetry Index" are strongly correlated with PC1, as indicated by
their long vectors. And also for the “Line Ratio” are strongly correlated with PC2.
Variables closer to the origin, such as "Hu Moment 7" and "Red Min" show weaker
correlations. Overall, PC1 and PC2 capture 39.19% of the total variance.
28
Data Analysis and AI Project 06/05/2024
To bee or not to bee
3) Other projections
In order to further our visual analysis, we decide to use other non-linear methods of
projection.
For starters, we apply the Multi Dimensional Scaling algorithm on our dataset.
As we can see, bees and bumblebees can be distinguished and while most others
cannot, this is not the case for butterflies whose data points are spread out from the
rest. This can be explained by the particular shape and colors of butterflies
compared to other bugs.
29
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Another non-linear projection we apply is the Isometric Mapping algorithm. We
choose various number of neighbors : 2, 10, 25 and 50 neighbors and here are the
results :
Similarly, bees, bumblebees and butterflies can be distinguished but that is not the
case with other bugs. Those shortcomings can be justified by the ISOMAP algorithm
being sensitive to noise.
30
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Finally, we apply the t-distributed stochastic neighbor embedding algorithm with
various levels of perplexity : 5, 25, 75 and 100.
Despite using various levels of perplexity, the bugs remain difficult to distinguish
using this algorithm.
From these various projections, we notice that while most bug types are difficult to
distinguish, bees and bumblebees can still be separated from one another. This is
useful as they were the 2 main bug types of this project.
After using the features in visualization, we could now use them for Machine learning
and Deep learning.
31
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Part 3 Machine Learning and Deep Learning
First of all we made a heat map of the correlation matrix with all the features.
According to it, we decided to select the following features: 'Area-Perimeter
Ratio','Hu Moment 1','Pixel Ratio','Blue Median', 'Contour Perimeter','Hu Moment 2'
and 'Homogeneity'. these features will be used for the 2 supervised methods that are
neither deep learning nor ensemble learning, the supervised ensemble learning
method, the 2 clustering methods and supervised method of your choosing trained
over optimally auto-encoded features based.
We normalized our data so that each feature contributes equally (and has the same
scale).
In order to optimize our results and enhance the strength of our classification model,
we have grouped rare species such as hover fly, wasp, butterfly, and dragonfly into a
single category called 'other.' In fact we only have 7 recorded specimens for these
species, which is not enough for a precise and reliable classification. A small dataset
can lead to overfitting, where the model learns noise and specific details that do not
generalize well to new data. It also complicates the bias variance trade-off, making it
challenging to balance between high variance (overfitting) and high bias
(underfitting).
Also 80% of our dataset will be used for the training set and 20% for the test dataset.
Fig 34 : Correlation coefficients heatmap
32
Data Analysis and AI Project 06/05/2024
To bee or not to bee
● 2 supervised methods that are neither deep learning nor
ensemble learning:
Because of the high dimensionality of our dataset we decided to use Logistic
Regression and SVM method. In fact it’s more robust against the complexity of our
features where LVQ, KNN and Decision tree could fail. Trying different methods of
logistic regression we got better results with linear one and whatever the value for
max iteration is (100 OR 1000). However in general, we got better results using the
rbf kernel for svm method with a c equal to 1 (so that it’s not too flexible and not too
rigid with the error classification).
here are our results:
Logistic Regression classification report:
Fig 35 : Logistic regression Classification report
Fig 36 : SVM Classification report
33
Data Analysis and AI Project 06/05/2024
To bee or not to bee
The Logistic Regression model performed well with an overall accuracy of 82%. It
excelled particularly with the "Bumblebee" class, achieving a perfect recall of 100%,
meaning it identified all bumblebees correctly. However, it showed some limitations
in fully capturing the "Bee" category, missing about 30% of actual bees. The
precision was very high across all classes, especially notable in the "autres"
category at 100%.
The SVM model, using a RBF kernel, achieved an overall accuracy of 80%. It was
particularly effective in identifying bumblebees with a high recall of 100%, though it
tended to misclassify other classes as bumblebees. The model struggled more with
the "Bee" and "autres" categories, especially in terms of recall, indicating missed
cases in these categories. Precision was strong in the "autres" category, perfect at
100%, but overall performance was less balanced compared to Logistic Regression.
34
Data Analysis and AI Project 06/05/2024
To bee or not to bee
● 1 supervised ensemble learning method [II.2413]
Again because of the number of features, and the noise we used the random forest
method which is performant concerning the noisy data management. concerning the
n estimator value we have seen that the scores were proportional to the n estimator
value so we kept the default value (100)
Fig 37 : Random Forest Classification report
The Random Forest model achieved an overall accuracy of 76% in classifying
different bug categories. It showed a strong ability to recognize the "Bumblebee"
class with a high recall of 95%, indicating it successfully identified nearly all
bumblebees. However, precision was lower at 68%, suggesting some
overclassification of other bugs as bumblebees. Performance for the "Bee" category
was fairly balanced with a precision of 84% and a recall of 70%. The "autres"
category, had a perfect precision (100%), and a low recall of 43%, indicating the
model missed more than half of the actual cases in this category.
35
Data Analysis and AI Project 06/05/2024
To bee or not to bee
● At least 2 clustering methods [II.2413]
For the clustering methods we will use the K Means and hierarchical clustering
method because we know the number of clusters.
Fig 38 : K-means plot
As you can see, the best number of clusters are 3 or 5 according to the graph and
the lowest C index score is with 3 clusters. In addition we know we have 3 different
numbers of bug types therefore we will choose 3 clusters. And for the Hierarchical
(agglomerative) clustering we used the ward linkage method so that the clusters are
equal and balanced compared to other linkage types where we could see one
principal cluster and two small clusters (by testing).
Fig 39 : K-means and Hierarchical clustering
36
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 40 : K-means and hierarchical scores
As we can see from the graphs and the indexes. Both clustering methods have a low
silhouette score, indicating that the clusters are not very well-separated and are quite
close to each other.
Concerning the Davies-Bouldin indexes, K-Means performs slightly better than
Hierarchical clustering, implying that it has a slightly better cluster separation and
compactness.
Again, K-Means shows a higher score than Hierarchical, suggesting better cluster
quality in terms of separation and compactness.
37
Data Analysis and AI Project 06/05/2024
To bee or not to bee
● At least 1 supervised neural network using your own
features [IG.2411] :
First of all, we had to determine the best features for our supervised algorithm using
a neural network.
With a Random Forest Classifier model, we were able to identify the 10 best
variables, as they have the highest importance values according to the
characteristics of the Random Forest model.
Fig 41 : Most important features
For the supervised model using a neural network, we opted for a Multilayer
Perceptron (MLP), a type of artificial neural network. Specifically, we used
TensorFlow, an open-source machine learning library.
38
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 42 : MultiLayer Perceptron
We split the data into a training set with 80% of the population and a test set with
20% of the population, all randomly.
We set up the model with dense layers using ReLU activation, which introduces
non-linearity into the model. The first dense layer with 64 neurons captures
information among the input features.
Similarly, a second dense layer with 32 neurons and ReLU activation allows us to
capture more complex content and patterns.
Finally, the output layer, with the softmax function, converts the outputs into
probabilities, enabling the identification of the most probable class.
By combining these layers, the MLP model can learn complex relationships in the
data and produce accurate predictions for the classification task.
After identifying the 10 best variables, we set up a brute force system to determine
the best supervised model with a neuron brain. For 10 variables by testing all
possible possibilities ranging from 1 to 10 variables, we have 1023 possibilities:
C(n,r) = n!/r!(n r)!
n
Total = ∑ C(n,r)
r=1
After several hours of processing to test all possible models and achieve the best
accuracy, we reached an accuracy of 0.82.
39
Data Analysis and AI Project 06/05/2024
To bee or not to bee
The model includes the following variables as inputs: 'Area-Perimeter Ratio',
'Homogeneity', 'Blue Mean', 'Blue Median', and 'Correlation'.
We used this model for 3 classes of bug type, wish ‘Bee’, ‘Bumblebee’ and ‘Others’.
So in this “Others' ' class we have the Butterfly, Hoverfly, Wasp and Dragonfly. We
put these 4 bug types in one, because we don’t have a lot of train data for these
types.
Fig 43 : Classification report
The precision and recall for the "Bee" class are quite high at 82% and 90%,
respectively, leading to an F1-score of 86%, suggesting that the model is particularly
effective at identifying bees.
For the "Bumblebee" class, the precision is slightly lower at 75%, and the recall is
80%, resulting in an F1-score of 77%, which is still acceptable but shows room for
improvement.
The "Others'' class, however, presents significant challenges, with perfect precision
at 100% but a very low recall of 20%, leading to an F1-score of only 33%. This
indicates that while the model rarely misclassifies other types as "Others," it fails to
identify many true "Others," suggesting a high number of false negatives.
The macro average shows a disparity between precision 86% and recall 63%,
highlighting the model uneven performance across different classes. The weighted
average F1-score of 78% confirms that the model performance is generally
robust but needs improvement in handling minority classes.
40
Data Analysis and AI Project 06/05/2024
To bee or not to bee
● A supervised method of your choosing trained over
optimally auto-encoded features based on your extracted
features. You will detail your auto-encoder architecture
and training process [IG.2411] :
After several attempts with numerous algorithms, we chose to use the pre-trained
model “yolov8n-cls.pt” which is part of the YOLO DARKNET environment.
That is the architecture of YOLOV8 :
Fig 44 : YOLO architecture
41
Data Analysis and AI Project 06/05/2024
To bee or not to bee
We can see in this schema that YOLOV8 is composed of two main parts: the
Backbone and the Head.
Firstly, the Backbone is responsible for extracting features from the input image. He
has three main steps:
- Conv: the Convolution an operation that detects patterns in the image.
- C2f: A module that combines features extracted at different levels.
- SPPF: A layer for merging information from different parts of the image.
Just the P1 to P5 represent different resolutions of the image, with each level
progressively reducing the image size while increasing the depth of the extracted
features.
Secondly, the Head is responsible for using the features extracted by the Backbone
to perform object detection. It also has three main modules :
- Upsample: Increases the resolution of the features to merge them with
features from previous levels.
- Concat: Concatenates features from different levels.
- Detect: Predicts objects in the image using convolutions to produce bounding
box coordinates, object classes and the confidence scores.
YOLOV8 has five variants such as nano(n) for the minimum and large(x) for the
maximum, based on the number of parameters. In our case, we use the nano(n)
model because we don’t have a lot of images and parameters. so we use the model
“yolov8n”. In Particular, we use the classification model for the project, so we need to
specify in the name of the model the classification by adding the “-cls”.
Finally, we have the model “yolov8n-cls.pt”.
For the execution of the algorithm, we need to prepare the image database so that a
portion of the images are in a training folder and a validation folder. The distribution
is 75% of the images in the training folder and 25% in the validation folder.
That is, 185 images in the training folder and 65 images in the validation folder. The
images are classified in folders by type of insect into 6 classes: Bee, Bumblebee,
Butterfly, Dragonfly, Hoverfly, and Wasp. The same applies to the validation folder.
train/ (185 images)
├── Bee/ (86 images)
├── Butterfly/ (11 images)
├── Hover fly/ (6 images)
├── Wasp/ (6 images)
├── Dragonfly/ (1 images)
└── Bumblebee/ (75 images)
42
Data Analysis and AI Project 06/05/2024
To bee or not to bee
validation/ (65 images)
├── Bee/ (29 images)
├── Butterfly/ (4 images)
├── Hoverfly/ (3 images)
├── Wasp/ (3 images)
├── Dragonfly/ (1 images)
└── Bumblebee/ (25 images)
More specifically, the training folder is used to train the model. The model draws
inspiration, learns, and adjusts from these images to minimize error and adjust its
weights. The validation folder is used to validate the model during training. It checks
the model's performance on images it has not yet seen. It also helps adjust the
model's hyperparameters to achieve the best accuracy while avoiding overfitting,
which is overlearning that can lead to poor generalization and misleading
performance.
43
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Just before starting the model, we changed the images and especially zoomed in
on the images to focus only on the bugs and not on other noise like flowers or
grass. To zoom in on the bug and only the bug, we used the image's mask link to
identify the bug's location in the images.
We are lucky that we have a very good resolution of the images to maintain the
quality of the pixels.
.
Fig 45 : Original image
Fig 46 : Zoomed image
44
Data Analysis and AI Project 06/05/2024
To bee or not to bee
After we made this modification for each image of the train and validation folder we
can configure the model with the training and validation images as input. I set 40
epochs, which is a measure indicating the number of complete passes of the
learning algorithm through the entire dataset. I specified that the images should be
resized to 640 pixels in width and length so that all images are the same size and
quality.
After more than an hour of execution, here are the results obtained for our best
model:
Fig 47 : Model results for bug type 6 classes
First, we can see in the first graph the distribution of the loss over the entire training
data set with respect to the epochs. The curve decreases rapidly during the initial
epochs, then continues to decrease more slowly and eventually stabilizes at a very
low value. This indicates that the model is effectively learning the features of the
training data. The lowest and therefore most optimal value is 0.05015.
45
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Next, we have the distribution of the loss over the entire validation data set with
respect to the epochs. The validation loss decreases initially, which is a good sign,
but it shows significant fluctuations, especially after about 15-20 epochs. These
fluctuations might indicate that the model is starting to overfit the training data.
However, the general trend shows a decrease in loss, which is positive.
The third graph shows the accuracy of the percentage of correct predictions among
the most probable predictions on the training set. The accuracy increases rapidly
and stabilizes around 0.9 or 90% after about 10-15 epochs. It reaches a maximum
of 95.4% correct predictions at epoch 21. This indicates that the model is capable
of learning the features of the training data and achieves high accuracy.
The last graph shows the accuracy of the percentage of correct predictions among
the top five most probable predictions on the training set. The accuracy very quickly
reaches 1 or 100%, indicating that the model is almost always able to place the
correct answer among the top five predictions. This is completely normal for
well-trained models.
The confusion matrix provides information on the model's correct and incorrect
predictions with respect to the validation values:
Fig 48 : Confusion matrix Normalized
46
Data Analysis and AI Project 06/05/2024
To bee or not to bee
We can see from this matrix that 97% of the bees are correctly predicted as bees.
Similarly, the Bumblebee, Dragonfly, and Wasp species have 100% of their data
predicted correctly. However, there is a larger discrepancy with the Butterfly species,
where 75% of the data is predicted as Butterfly, but the remaining 25% is predicted
as Bee. Likewise, the Hoverfly class has 67% of its data predicted correctly and 33%
predicted as Bee.
Overall, the YOLOV8 model demonstrates very good general performance, with
slight confusion for certain insect classes.
With to the resources of YOLOV8, we have in the outputs some batches of images
that it processed during the model's execution with these parameters and features:
47
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 49 : YOLO modified images during process
We can see that it significantly alters the colorimetry of the images, rotates the
images to one side or the other, zooms in, and adds black rectangular shapes to
highlight areas of interest or, conversely, areas that the model should ignore.
Next, it does the same with a mosaic of images for the validation set, indicating its
prediction above each respective image. Here is an example:
Fig 50 : YOLO predictions for validation dataset
48
Data Analysis and AI Project 06/05/2024
To bee or not to bee
We can distinctly differentiate the various types of insects by their names and also by
their colors. Each color corresponds to a specific type of insect.
● Prediction phase with a batch of test images:
Now we can move on to the prediction phase with the test images ranging from 251
to 347. We will use the same process of transforming the images with their masks to
have only a zoomed-in image of the insect in question.
Fig 51 : Original test image
Fig 52 : Zoomed test image
Then, we will use the previously created model through YOLO. We will import these
weights. We will create a loop that allows us to predict the type of insect for each
image in the folder containing the zoomed-in images from the test image set.
49
Data Analysis and AI Project 06/05/2024
To bee or not to bee
After making these predictions, it will put all this information into a CSV file with one
column for the image name and another for the predicted bug type.
Here are some statistics from the predictions made on all the test images:
Fig 53 : Predictions between Bug types
50
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 54 : Average confidence predictions between Bug types
The results show very high confidences for the "Wasp" with 99%, "Bumblebee" with
98%, and "Bee" with 96% classes, indicating that the model identifies these insects
with high precision. However, for the "Butterfly" class with 89%, although the
confidence remains high, there is slight uncertainty.
The prediction for the "Hover fly" class with a confidence of 50% reveals significant
uncertainty, suggesting that the model struggles to differentiate this class from
others.
Here are some examples of model output after image prediction from the test
database:
51
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Fig 55 : YOLO outputs for test dataset
52
Data Analysis and AI Project 06/05/2024
To bee or not to bee
These results highlight the model robustness for certain classifications while also
pointing out areas requiring improvement, particularly for classes with lower
confidences. To enhance overall accuracy, it will be necessary to reinforce the model
training with additional data.
● Bonus :
We will move on to the species prediction phase for the test image dataset. For this,
we will use the same methodology as for the prediction of bug types. We create a
model with YOLOV8 by specifying the species names instead of the insect type
names. Here are the model results for species prediction:
Fig 56 : Model results for species
53
Data Analysis and AI Project 06/05/2024
To bee or not to bee
According to the model results, we can see that the Train loss decreases significantly
over the epochs, going from 3 to 0.2 for the last epoch. However, the Validation loss
decreases but is less noticeable than the Train loss.
We achieve an accuracy of 0.8, or 80.6% correct predictions, at epoch 37 for the top
one prediction. Thus, for the top five predictions, we have 94.4% correct predictions.
The model shows promising and satisfactory results for predicting insect species.
We can now test the model on the Test datatest :
This is the repartition of the species prediction by the bug type.
Fig 57 : Predictions between Species
We see a good repartition between the Bee and the Bumblebee. The must higher
prediction Specie class is the Bombus hortorum with 28 predictions. And Apis
mellifera with 24 predictions.
54
Data Analysis and AI Project 06/05/2024
To bee or not to bee
We have grouped by the bug type to have a visualization and comparison of the
distribution of the species.
Fig 58 : Predictions between Species group by Bug Types
If we compare these distributions with the model Bug Types 6 classes, it’s a very
good repartition, but he cannot identify the hover fly bug types.
So this model on the Species has good metrics but not enough than the model on
the bug types 6 classes, because we have many more classes in the Species so the
computation and the vision are more difficult to have sufficient predictions.
55
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Conclusion
Pollinator identification is important in science and nature and fortunately, our project
managed to reach this goal. Throughout this project, we managed to use given bug
images and masks and identify a number of features in order to differentiate them.
Those features revolve around the shape of the bugs but also the colors of the bugs.
Having these features, we now had a better idea of bug identifications. By applying
projection of the various features, we were able to have a good sense of bug
repartitions throughout our dataset.
This project highlights the strengths and limitations of different Supervised Learning
Methods, Ensemble Learning Method and clustering methods in bug types
classification. In fact, logistic regression and SVM provided quite good classification
performance and Random Forest (overall accuracy of 76%) effectively managed
noisy data and performed well for 'Bumblebee' with high recall, but showed some
misclassifications. Logistic regression performed well identifying 'Bumblebee' with
perfect recall but had difficulties with the 'Bee' category, missing about 30% of actual
bees while SVM (with an overall accuracy of 80%) was particularly effective for
'Bumblebee' but struggled with 'Bee' and 'other' categories, showing less balanced
performance compared to Logistic Regression. Moreover both clustering methods
offer good insights into the data structure, even if we saw that there were some
difficulties in cluster separation K-Means performed slightly better in terms of cluster
separation and compactness. Finally the logistic regression method has the highest
overall accuracy (about 82%). Finding features that better differentiate the bug types
could improve the accuracy of the several méthods.
Throughout the project development process, we encountered various facets of
machine learning. With fluidity and enthusiasm, we devised bug prediction processes
using different models. Particularly noteworthy was the YOLO DARKNET model,
which allowed us to establish a training and validation process with the base dataset
containing six classes. We partitioned the dataset into 75% for training and 25% for
validation. Achieving an accuracy of 95% affirmed our choice and configuration of
the model.
Subsequently, we tested our model on a separate test dataset to predict bug types.
We are highly satisfied with the results, as there was a fair distribution of predictions
among the two largest classes, bees and bumblebees (see CSV file). Additionally,
we successfully predicted bug species, achieving a commendable distribution among
the various species. It's worth noting that this model deals with more than six
classes, making it more complex and approximate in its results.
56
Data Analysis and AI Project 06/05/2024
To bee or not to bee
Sources
- Stackoverflow symmetry
- Geeksforgeeks longest line
- Pl@ntNet
- Data Analysis Labs
- IA Labs
- Basic classification: Classify images of clothing | TensorFlow Core
- Brief summary of YOLOv8 model structure · Issue #189
- A Guide to YOLOv8 in 2024 viso.ai
- Multi-layer perceptron (MLP-NN) basic Architecture. | Download Scientific Diagram
- RandomForestClassifier — scikit-learn 1.5.0 documentation
- 1.4. Support Vector Machines — scikit-learn 1.5.0 documentation
- 2.3. Clustering — scikit-learn 1.5.0 documentation
- Deep Learning and Entropy-Based Texture Features for Color Image Classification
- Texture Features Extraction
- Computer Vision-Hu Moments
57
