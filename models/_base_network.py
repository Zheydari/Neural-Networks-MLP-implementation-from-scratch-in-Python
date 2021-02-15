# Do not use packages that are not in standard distribution of python
# Used some ideas from my previous work. This is MY GITHUB: https://github.com/aabid0193/ML-from-Scratch/blob/master/regression.py
import numpy as np
class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        ## assisstance on keep dims and axis solutions https://stackoverflow.com/questions/43290138/softmax-function-of-a-numpy-array-by-row
        exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        sum_exp = np.sum(exp, axis=-1, keepdims=True)
        prob =  exp/sum_exp
        return prob
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################

        # Loss for one sample: 𝑙(𝑥𝑖,𝑦𝑖,𝑤)=−[𝑦𝑖⋅𝑙𝑜𝑔𝑃(𝑦𝑖=1|𝑥𝑖,𝑤)+(1−𝑦𝑖)⋅𝑙𝑜𝑔(1−𝑃(𝑦𝑖=1|𝑥𝑖,𝑤))]
        # Loss for many samples: 𝐿(𝑋,𝑦⃗ ,𝑤)=(1/ℓ)∑𝑙(𝑥𝑖,𝑦𝑖,𝑤)
        m = x_pred.shape[0]
        num_classes = x_pred.shape[1]

        y_oh = np.eye(num_classes)[y] # one hot encoding of y

        cross_entropy = y_oh*np.log(x_pred)
        cost = -np.sum(cross_entropy)
        cost = cost/m
        loss = np.squeeze(cost) # remove single deminsional entries from shape
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        ## TP + TN / (TP + FP + TN + FN)
        y_pred = np.argmax(x_pred, axis=1)
        acc = (y == y_pred).mean()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc        

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        '''
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        out = 1. / (1. + np.exp(-1. * X)) 
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################

        ## d/dx theta(x) = theta(x)(1-theta(x)) where theta = 1/(1 + e^-x)
        sigmoid = 1. / (1. + np.exp(-1. * x))
        ds = sigmoid*(1-sigmoid)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        '''
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        out = np.maximum(0, X)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        '''
        #############################################################################
        # TODO: Comput the gradient of ReLU activation                              #
        #############################################################################
        # if X <= 0: 
        #     out = 0
        # else:
        #     out = 1
        out = np.greater(X, 0).astype(int)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
