import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
        
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    pass
    depth, Height_image, Width_image = input_dim
    
    
    ###Conv Layer - Conv - Relu - Max Pooling
    self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, depth, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)

    
    #Intermediate Hidden Layer : Affine - Relu Layer     
    input_dim_intermediate_layer = np.prod((num_filters,  Height_image/2, Width_image/2 ))
    
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(input_dim_intermediate_layer  , hidden_dim) )     
    self.params['b2'] = np.zeros(hidden_dim)
    
    #Output Layer: Affine - SoftMax Layer
    self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes) )     
    self.params['b3'] = np.zeros(num_classes)     
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    pass
    
    #Conv - Relu - Max_pooling
    conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    #conv_out = conv_out.reshape(conv_out.shape[0],-1) #Each row represents an example
    af_out, af_cache = affine_relu_forward(conv_out, W2, b2)

    scores, output_cache = affine_forward(af_out, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    pass
    loss, grad_softmax = softmax_loss(scores, y)
    
    #Adding Regularization to the Loss
    loss += self.reg * 0.5 * np.sum(W1**2) # L2 Regularization for W1
    loss += self.reg * 0.5 * np.sum(W2**2) # L2 Regularization for W2
    loss += self.reg * 0.5 * np.sum(W3**2) # L2 Regularization for W3
    ############################################################################
    
    ##Gradients w.r.t W1, W2 and W3.
    dout, dW3, db3 = affine_backward(grad_softmax, output_cache)
    dout, dW2, db2 = affine_relu_backward(dout, af_cache)
    dout, dW1, db1 = conv_relu_pool_backward(dout, conv_cache)
    
    ##################################
    #Saving and/or Regularizing the Gradients    
    #################################
    grads['W1'] = dW1 + self.reg * W1
    grads['W2'] = dW2 + self.reg * W2
    grads['W3'] = dW3 + self.reg * W3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    ###################################
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass



class MultiLayerConvNet(object):    
    
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32, N=1, M=1, use_batchnorm=False):
        """
    #Initialize a new network
    
    Architecture:  [conv-relu-pool]xN - conv - relu - [affine]xM - affine -  [softmax or SVM]

    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: List of Number of filters to use in the convolutional layers
    - filter_size: List of size of filters to use in the convolutional layer
    - hidden_dim: List of Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.N = N
        self.M = M
        
        depth, Height_image, Width_image = input_dim
        


        ############################################
        ###### Conv - Relu - Max Pooling ###########
        ############################################
        for i in range(self.N):    
      
            
            weight_name = 'W' + str(i+1)  ##Weight Names i.e. W1, W2, W3 ... WN
            bias = 'b' + str(i+1)
            
            
            self.params[weight_name] = np.random.normal(scale=weight_scale, size=(num_filters, depth, filter_size, filter_size))
            self.params[bias] = np.zeros(num_filters)        
            
            
            Height_image /= 2 
            Width_image /= 2
            
            depth = num_filters
        
        
        ###########################################
        ############ Conv - Relu ##################
        ###########################################
        weight_name = 'W' + str(self.N+1)
        bias = 'b' + str(self.N+1)
        
        self.params[weight_name] = np.random.normal(scale=weight_scale, size=(num_filters, depth, filter_size, filter_size))
        self.params[bias] = np.zeros(num_filters)
        
        
        depth = num_filters        
        ############################################   
        
        #############################################
        ################## Affine###################
        #############################################
        input_dim_hidden = np.prod ( (Height_image * Width_image * depth) )#Input dimensions of Hidden Layer
        output_dim_hidden = hidden_dim
        for i in range(self.M):
            weight_name = 'W' + str(i+1 + self.N+1)
            bias = 'b' + str(i+1 + self.N+1)
            
            
            self.params[weight_name] = np.random.normal(scale=weight_scale, size=(input_dim_hidden , output_dim_hidden) )            
            self.params[bias] = np.zeros(output_dim_hidden)
            
            input_dim_hidden = output_dim_hidden #Keeping the hidden dim constant for the other layers
            output_dim_hidden = output_dim_hidden
        
        ################################################# 
        ############## Affine SoftMax ###################
        #################################################
        
        
        weight_name = 'W' + str( self.N+1 + self.M+1) #Incrementing to all previous layers
        bias = 'b' + str(self.N+1 + self.M+1 ) #Incrementing to all previous layers
        
        self.params[weight_name] = np.random.normal(scale=weight_scale, size=(input_dim_hidden, num_classes) )
        self.params[bias] = np.zeros(num_classes)
        
        #################################################       
        ####Change the data types of all the initialized parameters
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        #####################################################################################
        ####################################################################################


  def loss(self, X ,y=None):
        """"
        For evaluating the Loss and Gradient :
        
        If Y = None:
        compute the forward pass and return the output of class scores
        else
        compute the backward pass as well and return the loss and gradient
        """
         
        #Some important calculations
        filter_size = self.params['W1'].shape[2]
        
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        pool_param = { 'pool_height': 2, 'pool_width': 2, 'stride':2 }
        
        
        N = self.N
        M = self.M
        
        
        #########################################################################
        #####################Computing the forward pass#######################
        ########################################################################

        ################################################
        ###### Conv - Relu - Max Pooling ###############
        ################################################
        conv_relu_pool_list = [] # List for storing the cache
        
        data_points = X.copy()
        
        for i in range(N):
            w = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)]
            data_points, cache = conv_relu_pool_forward(data_points, w, b, conv_param, pool_param)
            
            conv_relu_pool_list.append(cache)
            
        #####################################################
        ############# Conv - Relu ###########################
        #####################################################
        conv_relu_list = []   
        w = self.params['W' + str(N+1)]
        b = self.params['b' + str(N+1)]
        
        data_points, cache = conv_relu_forward(data_points, w, b, conv_param)             
        conv_relu_list.append(cache)
        

        ########################################################
        ################Affine Layers ##########################
        ########################################################
 
        affine_list = []
        for i in range(M):
            w = self.params['W' + str(i+1 + N+1)]
            b = self.params['b' + str(i+1 + N+1)]
            data_points, cache = affine_forward(data_points, w, b)
            affine_list.append(cache)
            
        ########################################################
        ################Affine Softmax Layers ##################
        ########################################################
       
        w = self.params['W' + str(M+1 + N+1)]
        b = self.params['b' + str(M+1 + N+1)]

        scores, softmax_cache = affine_forward(data_points, w, b)
        
        ########################################################################
        
        #############################
        ##Return scores if y is given
        #############################
        
        if y is None:
            return scores
            
        ##Intializing the loss and grads    
        loss, grads = None, {}
        
        #########Calculate the loss function ################
        loss, softmax_grad = softmax_loss(scores, y)
        
        #################################################
        ###########Computing the Backward Pass ##########
        #################################################
        
        #################################################
        ########### Softmax Affine Layer ################
        #################################################
        der_data_points, dw, db = affine_backward(softmax_grad, softmax_cache)
        
        #Storing the derivative + Regularization w.r.t corresponding weights
        grads['W' + str(M+1 + N+1)] = dw + self.reg * self.params['W' + str(M+1 + N+1)]
        grads['b' + str(M+1 + N+1)] = db
        
        #Adding the regularization to the loss w.r.t corresponding weights
        loss += 0.5 * self.reg * (self.params['W' + str(M+1 + N+1)] **2).sum()
        
        #################################################
        ############# Affine Layer ######################
        #################################################
        
        for i in range(M-1, -1, -1):
            der_data_points, dw, db = affine_backward(der_data_points, affine_list.pop())
        
            #Storing the derivative + Regularization w.r.t corresponding weights
            grads['W' + str(i+1 + N+1)] = dw + self.reg * self.params['W' + str(i+1 + N+1)]
            grads['b' + str(i+1 + N+1)] = db
        
            #Adding the regularization to the loss w.r.t corresponding weights
            loss += 0.5* self.reg * (self.params['W' + str(i+1 + N+1)] **2).sum()
        
        #################################################
        ############# Conv - Relu  ######################
        #################################################
                                 
        der_data_points, dw, db = conv_relu_backward(der_data_points, conv_relu_list.pop())
        
        #Storing the derivative + Regularization w.r.t corresponding weights
        grads['W' + str(N+1)] = dw + self.reg * self.params['W' + str(N+1)]
        grads['b' + str(N+1)] = db
                                 
        #Adding the regularization to the loss w.r.t corresponding weights
        loss += 0.5* self.reg * (self.params['W' + str(N+1)] **2).sum()
        
        #################################################
        ############# Conv - Relu - Pool ################
        #################################################
        for i in range(N-1, -1, -1):
            der_data_points, dw, db = conv_relu_pool_backward(der_data_points, conv_relu_pool_list.pop())
            
            #Storing the derivative + Regularization w.r.t corresponding weights
            grads['W' + str(i+1)] = dw + self.reg * self.params['W' + str(i+1)]
            grads['b' + str(i+1)] = db
                                 
            #Adding the regularization to the loss w.r.t corresponding weights
            loss += 0.5 * self.reg * (self.params['W' + str(i+1)] **2).sum()                         
        
                                     
                                     
        ##Return Loss and gradients
        return loss, grads
        ##############################################
        
 

class MultiLayerConvNet_2(object):    
    
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=7, hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, dtype=np.float32, N=0, M=0, use_batchnorm = False ):
        """
    #Initialize a new network
    
    Architecture:  [conv-relu-conv-relu-pool]xN - [affine]xM - affine -  [softmax or SVM]

    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: List of Number of filters to use in the convolutional layers
    - filter_size: List of size of filters to use in the convolutional layer
    - hidden_dim: List of Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.N = N*2
        self.M = M
        self.use_batchnorm = use_batchnorm
        
        depth, Height_image, Width_image = input_dim
        
        ###############################################################
        #############Intializations are made using ####################
        ##############Delving Deep into Rectifiers:####################
        #Surpassing Human-Level Performance on ImageNet Classification#
        ###############################################################
        
        ##########################################################
        ###### Conv - Relu - Conv - Relu - Max Pooling ###########
        ##########################################################
        filter_index = 0 
        for i in range(0, self.N, 2):    
      
            #################
            ###Conv - Relu###
            #################
            
            weight_name = 'W' + str(i+1)  ##Weight Names 
            bias = 'b' + str(i+1)
            
            input_size = np.prod( (depth, filter_size, filter_size) ) 
            
            
            self.params[weight_name] = np.random.normal(scale=np.sqrt(2.0/input_size) , size=(num_filters[filter_index], depth, filter_size, filter_size))
            self.params[bias] = np.zeros(num_filters[filter_index])        
            
            if self.use_batchnorm:
                gamma = 'gamma' + str(i/2+1)
                beta = 'beta' + str(i/2+1)
                self.params[gamma] = np.ones(num_filters[filter_index])        
                self.params[beta] = np.zeros(num_filters[filter_index])
            
            
            ##############################
            ###Conv - Relu - Pool #######
            ##############################
            weight_name = 'W' + str(i+2)  ##Weight Names 
            bias = 'b' + str(i+2)            
            #Num of Filters in Previous Layers will be the  depth of current input 
            depth = num_filters[filter_index]
            
            input_size = np.prod( (depth, filter_size, filter_size) ) 
            
            self.params[weight_name] = np.random.normal(scale=np.sqrt(2.0/input_size)  , size=(num_filters[filter_index], depth, filter_size, filter_size))           
            self.params[bias] = np.zeros(num_filters[filter_index])
                 
              
        
            Height_image /= 2 
            Width_image /= 2
            
            depth = num_filters[filter_index]
            filter_index += 1
        
        ####
        #########################################
        ################## Affine###################
        #############################################
        input_dim = np.prod ( (Height_image * Width_image * depth) )#Input dimensions of Hidden Layer
        output_dim_hidden = hidden_dim
        for i in range(self.M):
            weight_name = 'W' + str(i+1 + self.N)
            bias = 'b' + str(i+1 + self.N)            
            
            self.params[weight_name] = np.random.normal(scale=np.sqrt(2.0/input_dim)  , size=(input_dim , output_dim_hidden) )            
            self.params[bias] = np.zeros(output_dim_hidden)
            
            if self.use_batchnorm:
                self.params['gamma' + str(i+1 + N) ] = np.ones(output_dim_hidden)
                self.params['beta' + str(i+1 + N ) ] = np.zeros(output_dim_hidden)
            
            
            input_dim = output_dim_hidden #Keeping the hidden dim constant for the other layers
            output_dim_hidden = output_dim_hidden
        
        ################################################# 
        ############## Affine SoftMax ###################
        #################################################
        
        
        weight_name = 'W' + str( self.N + self.M+1) #Incrementing to all previous layers
        bias = 'b' + str(self.N + self.M+1 ) #Incrementing to all previous layers
        
        self.params[weight_name] = np.random.normal(scale=np.sqrt(2.0/input_dim)  , size=(input_dim, num_classes) )
        self.params[bias] = np.zeros(num_classes)
        
        
        
        
        ##############################################################       
        ##############Initializing BatchNorm parameters#############       
        ############## for running mean and variance ###############   
        ##############################################################
        
        self.bn_params = []
        if self.use_batchnorm:
          self.bn_params = [{'mode': 'train'} for i in xrange(N + self.M )]
        #################################################       
        ####Change the data types of all the initialized parameters
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
        #####################################################################################
        ####################################################################################


  def loss(self, X ,y=None):
        """"
        For evaluating the Loss and Gradient :
        
        If Y = None:
        compute the forward pass and return the output of class scores
        else
        compute the backward pass as well and return the loss and gradient
        """
        X = X.astype(self.dtype)
        
 ###############
        N = self.N
        M = self.M
        mode = 'test' if y is None else 'train'

                #If Any Conv layer is included
        if N>0:            
            filter_size = self.params['W1'].shape[2]
        
            conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
            pool_param = { 'pool_height': 2, 'pool_width': 2, 'stride':2 }
        
        
        if self.use_batchnorm:            
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        #########################################################################
        #####################Computing the forward pass#######################
        ########################################################################

        ################################################
        ###### Conv - Relu - Max Pooling ###############
        ################################################
        conv_relu_pool_list = [] # List for storing the cache
        conv_relu_list = [] 
        conv_bn_relu_list = []
        conv_bn_relu_pool_list = []
        data_points = X.copy()
        
        for i in range(0,N,2):
            w = self.params['W' + str(i+1)]
            b = self.params['b' + str(i+1)] 
            
            
            #If BatchNormailzation
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i/2+1)]
                beta = self.params['beta' + str(i/2+1)]      
                bn_param = self.bn_params[i/2]
                data_points, cache  = conv_bn_relu_forward(data_points, w, b, conv_param, gamma, beta, bn_param)
                conv_bn_relu_list.append(cache)
            else: 
                data_points, cache = conv_relu_forward(data_points, w, b, conv_param)
                conv_relu_list.append(cache)
            
            ####################################################
            ########Conv Relu Pool ###########################
            ##################################################
            
            w = self.params['W' + str(i+2)]
            b = self.params['b' + str(i+2)]            
            
            #if self.use_batchnorm:
            #    gamma = self.params['gamma' + str(i+2)]
            #    beta = self.params['beta' + str(i+2)]
            #    bn_param = self.bn_params[i+1]
            #    data_points, cache  = conv_bn_relu_pool_forward(data_points, w, b, conv_param, pool_param, gamma, beta, bn_param)
            #    conv_bn_relu_pool_list.append(cache)
            #else:             
            data_points, cache = conv_relu_pool_forward(data_points, w, b, conv_param, pool_param)            
            conv_relu_pool_list.append(cache)
            
        ########################################################
        ################Affine Layers ##########################
        ########################################################
 
        affine_list = []
        affine_bn_relu_list = []
        for i in range(M):
            w = self.params['W' + str(i+1 + N)]
            b = self.params['b' + str(i+1 + N)]
            
            if self.use_batchnorm:
                gamma = self.params['gamma' + str(i+1 + N/2)]
                beta = self.params['beta' + str(i+1 + N/2)]
                bn_param = self.bn_params[i + N/2]
                data_points, cache = affine_bn_relu_forward(data_points, w, b, gamma, beta, bn_param)
                affine_bn_relu_list.append(cache)
            else:    
                data_points, cache = affine_relu_forward(data_points, w, b)
                affine_list.append(cache)
            
        ########################################################
        ################Affine Softmax Layers ##################
        ########################################################
       
        w = self.params['W' + str(M+1 + N)]
        b = self.params['b' + str(M+1 + N)]
        
        scores, softmax_cache = affine_forward(data_points, w, b)
        
        ########################################################################
        
        #############################
        ##Return scores if y is given
        #############################
        
        if y is None:
            return scores
            
        ##Intializing the loss and grads    
        loss, grads = None, {}
        
        #########Calculate the loss function ################
        loss, softmax_grad = softmax_loss(scores, y)

        #################################################
        ###########Computing the Backward Pass ##########
        #################################################
        
        #################################################
        ########### Softmax Affine Layer ################
        #################################################
        der_data_points, dw, db = affine_backward(softmax_grad, softmax_cache)
        
        #Storing the derivative + Regularization w.r.t corresponding weights
        grads['W' + str(M+1 + N)] = dw + self.reg * self.params['W' + str(M+1 + N)]
        grads['b' + str(M+1 + N)] = db
        
        #Adding the regularization to the loss w.r.t corresponding weights
        loss += 0.5 * self.reg * (self.params['W' + str(M+1 + N)]**2).sum()
        
        #################################################
        ############# Affine Layer ######################
        #################################################
        
        for i in range(M-1, -1, -1):                
 
            
            if self.use_batchnorm:                
                der_data_points, dw, db, gamma, beta = affine_bn_relu_backward(der_data_points, affine_bn_relu_list.pop() )
                grads['gamma' + str(i+1 + N/2)] = gamma 
                grads['beta' + str(i+1 + N/2)] = beta                 
            else:    
                der_data_points, dw, db = affine_relu_backward(der_data_points, affine_list.pop())        
            
            #Storing the derivative + Regularization w.r.t corresponding weights
            grads['W' + str(i+1 + N)] = dw + self.reg * self.params['W' + str(i+1 + N)]
            grads['b' + str(i+1 + N)] = db
            
            
            
            #Adding the regularization to the loss w.r.t corresponding weights
            loss += 0.5* self.reg * (self.params['W' + str(i+1 + N)]**2).sum()
        
        ########################################### #####################
        ############# Conv - Relu - Conv - Relu -  Pool ################
        ################################################################
        for i in range(N-1, 0, -2):
            
            #if self.use_batchnorm:                
            #der_data_points, dw, db, gamma, beta = conv_bn_relu_pool_backward(der_data_points, conv_bn_relu_pool_list.pop())
            #grads['gamma' + str(i+1)] = gamma
            #grads['beta' + str(i+1)] = beta
            #else:                
            der_data_points, dw, db = conv_relu_pool_backward(der_data_points, conv_relu_pool_list.pop())
            
            #Storing the derivative + Regularization w.r.t corresponding weights
            grads['W' + str(i+1)] = dw + self.reg * self.params['W' + str(i+1)]
            grads['b' + str(i+1)] = db
                                 
            #Adding the regularization to the loss w.r.t corresponding weights
            loss += 0.5 * self.reg * (self.params['W' + str(i+1)]**2).sum()                         
            
            ###################################################
            ############Conv _ relu ##########################
            ######################## #############################
            #print i/2
            if self.use_batchnorm:
                der_data_points, dw, db, gamma, beta = conv_bn_relu_backward(der_data_points, conv_bn_relu_list.pop())               
                grads['gamma' + str(i/2 + 1 )] = gamma
                grads['beta' + str(i/2 + 1 )] = beta
            else:    
                der_data_points, dw, db = conv_relu_backward(der_data_points, conv_relu_list.pop())
            
            #Storing the derivative + Regularization w.r.t corresponding weights
            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db
                                 
            #Adding the regularization to the loss w.r.t corresponding weights
            loss += 0.5 * self.reg * (self.params['W' + str(i)]**2).sum()                         
        
            
                                     
                                     
        ##Return Loss and gradients
        return loss, grads
        ##############################################
        
    