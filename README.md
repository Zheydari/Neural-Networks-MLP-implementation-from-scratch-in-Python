# Neural Networks implementation from scratch in Python
A simple pipeline of training neural networks in Python to classify **Iris flowers** from petal and sepal dimensions (can also be used on any other multiclass classification dataset). Implemented two neural network architectures along with the code to load data, train, optimize these networks and classify the dataset. The main.py contains the major logic of this pipeline. You can execute it by invoking the following command where the yaml file contains all the hyper-parameters. 
<br>

```python
$: python main.py --config configs/config_file.yaml
```
<br>

There are three pre-defined config files under **./configs**. Two of them are default hyper-parameters for models (Softmax Regression and 2-layer MLP). 
Do **NOT** modify values in these config files. The third config file, **config_exp.yaml**, is used for your hyper-parameter tuning experiments and you are 
free to modify values of the hyper-parameters in this file. The script trains a model with the number of epochs specified in the config file. 
At the end of each epoch, the script evaluates the model on validation set. After the training completes, the script finally evaluate the best model on test data. 

**Python and dependencies** <br>

We will work with Python 3. If you do not have a python distribution installed yet, 
we recommend installing Anaconda (or miniconda) with Python 3. We provide **environment.yml** which contains a list of libraries needed to set environment 
for this implementation. You can use it to create a copy of conda environment. <br>

```python
$: conda env create -f environment yaml
```
<br>
If you already have your own Python development environment, please refer to this file to find necessary libraries.
<br>

## Data Loading

The IRIS dataset (iris_train.csv and iris_test.csv ) is present in the **./data** folder. 
<br>

**1.1 Data Preparation** 
<br>

To avoid the choice of hyper-parameters "overfits" the test data, it is a common practice to split the training dataset into the actual **training data** and **validation data** 
and perform hyper-parameter tuning based on results on validation data. Additionally, in deep learning, training data is often forwarded to models in **batches** for faster training time and noise reduction.
<br>

In our pipeline, we first load the entire data into the system, followed by a training/validation split on the training set. 
We simply use the first 80% of the training set as our training data and use the rest training set as our validation data. 
We have also organized our data (training, validation, and test) in batches and use different combination of batches
in different epochs for training data. 

## Model Implementation 

We now implement two networks from scratch: a simple **softmax regression** and a **two-layer multi-layer perceptron (MLP)**. 
Definitions of these classes can be found in **./models**. Weights of each model will be randomly initialized upon construction and stored in a weight dictionary. 
Meanwhile, a corresponding gradient dictionary is also created and initialized to zeros. 
Each model only has one public method called forward, which takes input of batched data and corresponding labels and returns the loss and accuracy of the batch.
Meanwhile, it computes gradients of all weights of the model (even though the method is called forward!) based on the training batch. 
<br>

**2.1 Utility Function** 
<br>
There are a few useful methods defined in **./_base network.py** that can be shared by both models. <br>

**(a) Activation Functions.** <br>
There are two activation functions used for this model: **ReLU** and **Sigmoid**. 
Implemented both functions as well as their derivatives in **./_base_network.py** (i.e, sigmoid, sigmoid_dev, ReLU, and ReLU_dev). 
<br>

**(b) Loss Functions.** 
<br>
The loss function used in this project is **Cross Entropy Loss**. Implemented both **Softmax function** and the computation of **Cross Entropy Loss** in **./_base_network.py**. <br>

**(c) Accuracy.** <br> 

We are also interested in knowing how our model is doing on a given batch of data. 
Therefore, we have implemented the **compute_accuracy** method in **./_base_network.py** to compute the accuracy of given batch. 
<br>

**2.2 Model Implementation**  
<br> 
The Softmax Regression is composed by a fully-connected layer followed by a ReLU activation. The two-layer MLP is composed by two fully-connected layers with a Sigmoid Activation in between followed by the softmax function before computing the loss (in both the models)! 
<br>

## Optimizer 

We will use an optimizer to update weights of models. An optimizer is initialized with a specific learning rate and a regularization coefficients. 
Before updating model weights, the optimizer applies **L2 regularization** on the model. Also implemented a vanilla **SGD optimizer**. 
<br>

***NOTE: Regularization is NOT applied on bias terms!***

## Visualization 
It is always a good practice to monitor the training process by monitoring the learning curves.
Our training method in main.py stores averaged loss and accuracy of the model on both training and validation data at the end 
of each epoch and plots the same.
<br>
