import numpy as np
from random import shuffle

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
  - f: A numpy array of shape (N, C) containing X.dot(W).

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

  '''
  num_train = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  for i in xrange(num_train):
    # Compute vector of scores
    f_i = X[i].dot(W)

    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    f_i -= np.max(f_i)

    # Compute loss (and add to it, divided later)
    sum_j = np.sum(np.exp(f_i))
    p = lambda k: np.exp(f_i[k]) / sum_j
    loss += -np.log(p(y[i]))

    # Compute gradient
    # Here we are computing the contribution to the inner sum for a given i.
    for k in range(num_classes):
      p_k = p(k)
      dW[:, k] += (p_k - (k == y[i])) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W

  '''
  N = X.shape[0]
  C = W.shape[1]
  for i in xrange(N):
      f_i = X[i].dot(W)
      f_i -= np.max(f_i)

      loss += -f_i[y[i]] + np.log(np.sum(np.exp(f_i)))

      for j in xrange(C):
          dW[:,j] += -X[i,:]*(j==y[i]) + np.exp(f_i[j])/np.sum(np.exp(f_i))*X[i,:]

  loss /= N
  dW /= N
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  '''
  N = X.shape[0]

  f = X.dot(W)
  p = np.exp(f - np.max(f, axis=1, keepdims=True))
  p /= np.sum(np.exp(f), axis=1, keepdims=True)

  loss = np.sum(-np.log(p[np.arange(N), y]))

  p[np.arange(N), y] -= 1
  dW = X.T.dot(p)

  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  dW /= N
  dW += reg*W
  '''
  loss = 0.0
  dW = np.zeros_like(W)
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Step 1: let's remove the numeric instability
  f = X.dot(W)
  # remove the max of each score column
  f_max = np.max(f).reshape(-1, 1)
  f -= f_max
  scores = np.exp(f)

  # Step 2: let's compute the loss
  # summing everything across the # of samples
  scores_sums = np.sum(scores, axis=1)
  # select all the valid scores
  scores_correct = scores[np.arange(N), y]
  f_correct = f[np.arange(N), y]
  loss = np.sum(-f_correct+np.log(scores_sums))

  # Step 3: let's compute the gradient of the function
  # We need to first take the scores of all cells - already done by scores
  # afterwards, we need to divide all of them row-wise by the sum of each row's scores
  sum = scores/(scores_sums.reshape(-1,1))
  # later on, we're gonna need a binary matrix for adding the 1's inside of the dW[:,j]
  bi_matrix = np.zeros_like(scores)
  bi_matrix[np.arange(N), y] = -1

  # Then, recall we need to either add 1 or subtract 1 to each element if it's in the correct class
  sum += bi_matrix

  # Then, we will multiply it elementwise by X_i(this is kind of weird) to get a 3D array of NxDxC
  dW = (X.T).dot(sum)

  # Don't forget the regularization
  loss /= N
  loss += reg*np.sum(W**2)/2
  dW /= N
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
