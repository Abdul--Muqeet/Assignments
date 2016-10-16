from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def affine_bn_forward(x, w, b, gamma, beta, bn_param):
    """
  A convenience layer that perform convolutional layaer followed by batch_normalization 

  Inputs:
  - x: Input to the affine  layer
  - w, b, bn_param: Weights and parameters for the affine layer 
  - bn_param: Object for tracking means and variance for batch norm
  - gamma: Scale parameter
  - beta: shift parameter
  
  
  Returns a tuple of:
  - out: Output from the Batch Norm
  - cache: Object to give to the backward pass
  """
    a, fc_cache = affine_forward(x, w, b)
    out, batch_cache =  batchnorm_forward(a, gamma, beta, bn_param)      
    cache = (fc_cache, batch_cache)
    return out, cache

def affine_bn_backward(dout, cache):
    """
  Backward Pass for affine - batch norm layers
  
  """

    fc_cache, bn_cache  = cache    
    da, gamma, beta = batchnorm_backward(dout, bn_cache)  
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, gamma, beta


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
  A convenience layer that perform convolutional layer followed by batch_normalization and then RELU non-linearity:

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param, bn_param: Weights and parameters for the convolutional layer and batch  
  - bn_param: Object for tracking means and variance
  - gamma: Scale parameter
  - beta: shift parameter
  
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
    a, fc_cache = affine_forward(x, w, b)
    a, batch_cache =  batchnorm_forward(a, gamma, beta, bn_param)  
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, batch_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout, cache):
    """
  Backward Pass for affine - batch norm - relu 
  
  """

    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    da, gamma, beta = batchnorm_backward(da, bn_cache)  
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, gamma, beta

def conv_bn_relu_forward(x, w, b, conv_param,  gamma, beta, bn_param):
  """
  A convenience layer that performs a convolution followed by batchnorm and then ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - gamma, beta, bn_param: Scale and Shift parameters for batch normailzation
  
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  
  a, bn_cache =  spatial_batchnorm_forward(a, gamma, beta, bn_param)  
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  """
  Backward pass for the conv - bn - relu- convenience layer
  """
  conv_cache, bn_cache, relu_cache = cache
  
  
  da = relu_backward(dout, relu_cache)
  ds, gamma, beta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(ds, conv_cache)
  return dx, dw, db, gamma, beta


def conv_bn_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer  
  - pool_param: Parameters for the pooling layer
  - gamma, beta, bn_param: Shift and Scale parameters for the Convolutional layers
  

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  a, bn_cache =   spatial_batchnorm_forward(x, gamma, beta, bn_param)  
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-batch_norm_relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  da, gamma, beta = spatial_batchnorm_backward(da, bn_cache)  
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, gamma, beta