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

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  examples = X.shape[0]
  classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  gradScore = np.zeros((examples,classes),float) #############################################################################
  pass 
  for i in range(examples):
    scores = X[i].dot(W)
    scores -= np.max(scores)
  # get unnormalized probabilities
    exp_scores = np.exp(scores)
    # normalize them for each example
    prob = exp_scores / np.sum(exp_scores, keepdims=True)        
    gradScore[i,:] = prob
    gradScore[i,y[i]] -= 1
    
    loss += - np.log(prob[y[i]])   
 
  examples = X.shape[0]
  gradScore /= examples
  dW =  np.dot(X.T,gradScore)  
  
  loss = np.sum(loss)/(examples) 

  loss += 0.5*reg * np.sum(W*W)
  dW += reg * W
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
  pass
  classes = W.shape[1]
  train_examples = X.shape[0]

  scores = np.dot(X, W)   
  scores -= np.max(scores,axis=1,keepdims=True)
  expScores = np.exp(scores)
  prob = (expScores)/np.sum(expScores,axis=1,keepdims=True) #NXC  
  loss += np.sum(- np.log(prob[range(train_examples),y]))

  dscores = prob
  dscores[range(train_examples),y] -= 1
  dscores /= train_examples  

  dW  = np.dot(X.T,dscores)
  dW += reg*W # regularization gradient
  
#Loss for all examples

  #Average Loss
  loss = loss/train_examples

  loss += 0.5 * reg * np.sum(W*W)

      
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

