from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    #epsilon = sqrt(6) / sqrt(n_in + n_out);
    #W = (np.random.rand(n_out, n_in)*2* epsilon) - epsilon;
    #W[:, -1] = 1

    return W
    
def forwardPass(W1, W2, data):
	W1Trans = np.transpose(W1)
	hidden = np.dot(data, W1Trans)
	hidden = sigmoid(hidden)

	# setting the hidden bias node value to 1 directly
	bias = np.ones((data.shape[0]))
	hidden = np.column_stack((hidden, bias))


	W2Trans = np.transpose(W2)
	output = np.dot(hidden, W2Trans)
	output = sigmoid(output)

	return hidden, output

def backwardPass(input_data, input_label, W1, W2, act3, act2, lamdaval):
	label_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	label_vec = np.tile(label_vec, np.size(input_label))
	label_vec = np.reshape(label_vec, (np.size(input_label), 10))

	for i in range(0, label_vec.shape[0]):
		np.add.at(label_vec[i], input_label[i], 1)

	obj_val = (label_vec-act3)
	obj_val = (1/2) * np.sum(np.power(obj_val, 2)) / (input_data.shape[0])
	#obj_val = np.sum(obj_val) / input_data.shape[0]
	reg = ((lambdaval * (np.sum(np.power(W1, 2)) + np.sum(np.power(W2, 2))))/ (2 * input_data.shape[0]))
	obj_val += reg

	errorAtOutput = np.array([])
	errorAtOutput = (act3-label_vec) * (act3) * (1 - act3)
	dJdW2 = np.dot(np.transpose(errorAtOutput), act2)

	errorHidden = np.array([])
	step1 = np.dot(errorAtOutput, W2)
	errorHidden = step1 * (act2) * (1 - act2)
	#dJdW1 = np.dot(np.transpose(input_data), errorHidden)
	dJdW1 = np.dot(np.transpose(errorHidden), input_data)

	dJdW1= np.delete(dJdW1,dJdW1.shape[0]-1,0)	
	#Regularization
	dJdW1 = (dJdW1 + (lambdaval * W1)) / input_data.shape[0]
	dJdW2 = (dJdW2 + (lambdaval * W2)) / input_data.shape[0]

	#dJdW1[:,-1] = 0
	#dJdW2[:,-1] = 0

	obj_grad = np.concatenate((dJdW1.flatten(), dJdW2.flatten()),0)

	#print (obj_val)

	return (obj_val, obj_grad)

def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    
    return  1.0 / (1.0 + np.exp(-1.0 * np.array(z))) #your code here
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    #mat = loadmat('C:\\Users\\debik\\Documents\\UB\\2nd Sem\\CSE574\\basecode\\basecode\\mnist_all.mat') #loads the MAT object as a Dictionary
    mat = loadmat("mnist_all.mat")
    
    #Your code here
    train_data = np.array([]).astype('float').reshape(0, 784)
    train_label = np.array([]).astype('float').reshape(0, 10)
    validation_data = np.array([]).astype('float').reshape(0, 784)
    validation_label = np.array([]).astype('float').reshape(0, 10)
    test_data = np.array([]).astype('float').reshape(0, 784)
    test_label = np.array([]).astype('float').reshape(0, 10)

    for i in range(0, 10):
        train_data = np.vstack((train_data, mat["train" + str(i)][0:5000]))
        train_label = np.append(train_label, np.repeat(i, mat["train" + str(i)][0:5000].shape[0]))
        validation_data = np.vstack((validation_data, mat["train" + str(i)][5000:]))
        validation_label = np.append(validation_label, np.repeat(i, mat["train" + str(i)][5000:].shape[0]))
        test_data = np.vstack((test_data, mat["test" + str(i)]))
        test_label = np.append(test_label, np.repeat(i, mat["test" + str(i)].shape[0]))

    columns_del_array=np.array([])
    for i in range(0, train_data.shape[1]):
        if (np.size(np.unique(train_data[:,i])) <= 1):
            columns_del_array = np.append(columns_del_array, i)

    #for i in columns_del_array:
     #   train_data = np.delete(train_data, i, 1)
      #  test_data = np.delete(test_data, i, 1)
        
    train_data=train_data/255
    validation_data=validation_data/255
    test_data=test_data/255


    #for i in range(0, validation_data.shape[1]):
     #   if (np.size(np.unique(validation_data[:,i])) < 5):
     #       columns_del_array = np.append(columns_del_array, i)

 #   for i in columns_del_array:
  #      validation_data = np.delete(validation_data, i, 1)
 
   # columns_del_array=np.array([])
    #for i in range(0, test_data.shape[1]):
       # if (np.size(np.unique(test_data[:,i])) < 5):
        #    columns_del_array = np.append(columns_del_array, i)

    #for i in columns_del_array:
     #   test_data = np.delete(test_data, i, 1)

    bias = np.ones((train_data.shape[0]))
    train_data = np.column_stack((train_data, bias))

    bias = np.ones((validation_data.shape[0]))
    validation_data = np.column_stack((validation_data, bias))

    bias = np.ones((test_data.shape[0]))
    test_data = np.column_stack((test_data, bias))

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    #w1 = params[0:n_hidden * (n_input)].reshape( (n_hidden, (n_input)))
    #w2 = params[(n_hidden * (n_input)):].reshape((n_class, (n_hidden)))

    act2, act3 = forwardPass(w1, w2, training_data)

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    #obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_val = 0
    obj_grad = np.array([])

    obj_val, obj_grad = backwardPass(training_data, training_label, w1, w2, act3, act2, lambdaval)
    
    return (obj_val,obj_grad)


def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here

    hidden, output = forwardPass(w1, w2, data)

    labels = np.argmax(output,1) 	

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]-1; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.3;

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#w1 = nn_params.x[0:n_hidden * (n_input)].reshape( (n_hidden, (n_input)))
#w2 = nn_params.x[(n_hidden * (n_input)):].reshape((n_class, (n_hidden)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

f = open("params.pickle", "ab")
pickle.dump(n_hidden, f)
pickle.dump(initial_w1, f)
pickle.dump(initial_w2, f)
pickle.dump(lambdaval, f)
f.close()

#f = open("params.pickle", "rb")
#an_hidden = pickle.load(f)
#aw1 = pickle.load(f)
#aw2 = pickle.load(f)
#alambdaval = pickle.load(f)
#f.close()
