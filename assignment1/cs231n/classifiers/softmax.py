from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # forward
    for i in range(N):
        for j in range(C):
            out[i, j] = X[i, :] @ W[:, j]
            out[i, :] = np.exp(out[i, :])
            out[i, :] /= np.sum(out[i, :])
    
    # loss
    loss -= np.sum(np.log(out[np.arange(N), y]))
    loss /= N
    loss += .5 * reg * np.sum(W * W)

    # backward
    out[np.arange(N), y] -= 1

    for i in range(D):
        for j in range(C):
            dW[i, j] = X[:, j] @ out[:, j].T

    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # forward
    out = X @ W
    out = np.exp(out)
    out /= np.sum(out, axis=1, keepdims=True)
  
    # loss
    loss -= np.sum(np.log(out[np.arange(N), y]))
    loss /= N
    loss += .5 * reg * np.sum(W * W)

    # backward
    dout = np.copy(out)
    dout[np.arange(N), y] -= 1
    dW = X.T @ dout
    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
