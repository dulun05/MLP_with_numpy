"""
Implementation of layer functions.
"""

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine transformation function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    
    Seems like (d_1, ..., d_k) has already been transformed into D = d_1 * ... * d_k here.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """

    # Implement the affine forward pass. Store the result in 'out'.            #
    N = x.shape[0]
    out = np.array(np.mat(x) * np.mat(w)) + np.array(np.mat(np.ones(N)).T * np.mat(b))

    #raise NotImplementedError
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine transformation function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """

    # Implement the affine backward pass.                                      #
    N = x.shape[0]
    dx = np.array(np.mat(dout) * np.mat(w).T) 
    dw = np.array(np.mat(x).T * np.mat(dout))
    db = np.array(np.mat(dout).T * np.mat(np.ones(N)).T).reshape(dout.shape[1])
    #raise NotImplementedError

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs) activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    #       Implement the ReLU forward pass.                                   #
    out = np.maximum(x, 0)
    #raise NotImplementedError

    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs) activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    #       Implement the ReLU backward pass.                                  #
    dx = dout * ((relu_forward(x) > 0)*1)

    #raise NotImplementedError
    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x))

    Inputs:
    - x: (float) a tensor of shape (N, #classes); so x already means X(N, D+1)*W(D+1, K) here? -Lun
    - y: (int) ground truth label, a array of length N

    

    Returns:
    - loss: the cross-entropy loss
    - dx: gradient of loss wrt input x
    """
    # stability: to avoid log(0) when calculating the cross entropy, add a small number to it, log(0+epsilon)
    epsilon = 1e-15

    from .classifiers.softmax import softmax, onehot, cross_entropy
    # calculate cross entropy loss and gradients                             #

    N, K = x.shape
    L = np.zeros((N, ), dtype=np.float32)

    sigma = softmax(x)
    P = onehot(y, K)
    L = cross_entropy(P, sigma+epsilon)
    loss = L.sum() / N
    dx =  np.array(np.mat(sigma) - np.mat(P)) / N
    #raise NotImplementedError
    return loss, dx


def check_accuracy(preds, labels):
    """
    Return the classification accuracy of input data.

    Inputs:
    - preds: (float) a tensor of shape (N,)
    - y: (int) an array of length N. ground truth label 
    Returns: 
    - acc: (float) between 0 and 1
    """

    return np.mean(np.equal(preds, labels))
