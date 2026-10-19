## Homework 10

In this homework, you are going to use the code from ```TrafficSignsClassification``` notebook and create your own traffic sign classifier.

### Step 1
Use the data from ```data/subset_homework``` folder and visualize some examples. How many images are there for each class?

### Step 2
Run the training with one single neuron (as we did in the lecture). What accuracy can you achieve?

### Step 3
Make further modifications to improve the accuracy (e.g. add more neurons, more layers, etc.). What is the maximum accuracy you can achieve?


So I've made so tests with models and number of epochs to train, as a result the most accuracy was achieved is 0.9984340044742729

It is a model with 4 hidden layers and that was trained for 50 epochs
But in general trained model have accuracy > 0.9

Here are results
* ACCURACIES FOR 10 EPOCHS:  [0.9196868008948546, 0.9525727069351231, 0.9588366890380313, 0.9738255033557047, 0.943400447427293]  MAX ACCURACY IS MODEL WITH  3  LAYERS, VALUE:  0.9738255033557047

* ACCURACIES FOR 20 EPOCHS:  [0.9521252796420582, 0.9659955257270694, 0.974496644295302, 0.9805369127516779, 0.9894854586129754]  MAX ACCURACY IS MODEL WITH  4  LAYERS, VALUE:  0.9894854586129754
 
* ACCURACIES FOR 30 EPOCHS:  [0.9671140939597316, 0.9778523489932885, 0.991275167785235, 0.992393736017897, 0.9901565995525727]  MAX ACCURACY IS MODEL WITH  3  LAYERS, VALUE:  0.992393736017897

* ACCURACIES FOR 40 EPOCHS:  [0.959731543624161, 0.9838926174496644, 0.9914988814317673, 0.9863534675615212, 0.9901565995525727]  MAX ACCURACY IS MODEL WITH  2  LAYERS, VALUE:  0.9914988814317673

* ACCURACIES FOR 50 EPOCHS:  [0.9718120805369127, 0.9874720357941834, 0.9903803131991051, 0.9847874720357942, 0.9789709172259508]  MAX ACCURACY IS MODEL WITH  2  LAYERS, VALUE:  0.9903803131991051

As we can see accuracies close 0.97 or higher by combining more layers but not so many epochs to train like 3 layers and 10 epochs or 30 epochs with 3 layers to have accuracy 0.99
So we can say that the accuracy should be tested using different epochs and layers to train the modelm it should be tested manually.