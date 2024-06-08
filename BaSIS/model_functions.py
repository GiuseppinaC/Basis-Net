
###################################################################
# BASIS-Net 
# From Point Estimate to Predictive Distribution in Neural Networks 
# - A Bayesian Sequential Importance Sampling Framework        
###################################################################

import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import time, sys
import pickle
import timeit
import scipy
from scipy import stats
import random
from random import choice
from random import seed
plt.ioff()
mnist = tf.keras.datasets.mnist
import pandas as pd
from scipy.stats import norm
from mpl_toolkits import axes_grid1

class BaSIS_first_Conv(keras.layers.Layer):
    def __init__(self,   tau = 0.003, kernel_size=5, kernel_num=16, kernel_stride=1, padding="VALID"):
        super(BaSIS_first_Conv, self).__init__()
        
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        self.tau = tau
        
    def build(self, input_shape):
       
        self.w_mu = self.add_weight(name='w_mu', shape=(self.kernel_size, self.kernel_size,  input_shape[-1], self.kernel_num),
                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None),
                                    regularizer=tf.keras.regularizers.l2(self.tau),#l1_l2(l1=tau, l2=tau),                                    
                                    trainable=True,
                                    )
        
    def call(self, mu_in):
        mu_out = tf.nn.conv2d(mu_in, self.w_mu,  strides=[1,self.kernel_stride,self.kernel_stride,1], padding=self.padding, data_format='NHWC')
        return mu_out
    
        
class BaSIS_MaxPooling(keras.layers.Layer):
    """BaSIS_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='SAME'):
        super(BaSIS_MaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
    def call(self, mu_in): 
        mu_out, argmax_out = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, self.pooling_size, self.pooling_size, 1], strides=[1, self.pooling_stride, self.pooling_stride, 1], padding=self.pooling_pad) #shape=[batch_zise, new_size,new_size,num_channel]    
        return mu_out
        

class BaSIS_Conv(keras.layers.Layer):
    def __init__(self, tau =0.0001,   kernel_size=3, kernel_num=16, kernel_stride=1, mean_mu=0,mean_sigma=0.1, padding="VALID"):
        super(BaSIS_Conv, self).__init__()
        self.tau= tau
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        self.mean_mu =mean_mu
        self.mean_sigma=mean_sigma
        
    def build(self, input_shape):
        MYinitializer = tf.keras.initializers.TruncatedNormal(mean=self.mean_mu, stddev=self.mean_sigma)
        
        self.w_mu = self.add_weight(name='w_mu', shape=(self.kernel_size, self.kernel_size,  input_shape[-1], self.kernel_num),
                                    initializer=MYinitializer,
                                    regularizer=tf.keras.regularizers.l2(self.tau),#l1_l2(l1=tau, l2=tau),                                    
                                    trainable=True,
                                    )
        
    def call(self, mu_in):
        mu_out = tf.nn.conv2d(mu_in, self.w_mu,  strides=[1,self.kernel_stride,self.kernel_stride,1],padding=self.padding, data_format='NHWC')
        return mu_out
    
class BaSIS_Flatten_and_FC(keras.layers.Layer):   
    def __init__(self, units, tau):
        super(BaSIS_Flatten_and_FC, self).__init__()
        self.units = units
        self.tau =tau

                       
    def build(self, input_shape):
        
        self.w_mu = self.add_weight(name = 'w_mu', shape=(input_shape[1]*input_shape[2]*input_shape[-1], self.units),
            initializer=tf.random_normal_initializer( mean=0.0, stddev=0.05, seed=None), regularizer=tf.keras.regularizers.l2(self.tau),#l1_l2(l1=tau, l2=tau), 
            trainable=True,
        )
          
    def call(self, mu_in): 
        batch_size = mu_in.shape[0] #shape=[batch_size, im_size, im_size, num_channel]   
        mu_flatt = tf.reshape(mu_in, [batch_size, -1]) #shape=[batch_size, im_size*im_size*num_channel]           
        mu_out = tf.matmul(mu_flatt, self.w_mu) 
        return mu_out            

		
class softmax(keras.layers.Layer):
    """softmax"""
    def __init__(self):
        super(softmax, self).__init__()
    def call(self, mu_in):
        mu_out = tf.nn.softmax(mu_in)
        return mu_out

class BaSIS_ReLU(keras.layers.Layer):
    def __init__(self):
        super(BaSIS_ReLU, self).__init__()
    def call(self, mu_in):
        mu_out = tf.nn.relu(mu_in)    
        return mu_out  

class BaSIS_eLU(keras.layers.Layer):
    def __init__(self):
        super(BaSIS_eLU, self).__init__()
    def call(self, mu_in):
        mu_out = tf.nn.elu(mu_in)    
        return 
    
class BaSIS_Dropout(keras.layers.Layer):
    def __init__(self, drop_prop):
        super(BaSIS_Dropout, self).__init__()
        self.drop_prop = drop_prop

    def call(self, mu_in, Training=True):
        if Training:        
           mu_out = tf.nn.dropout(mu_in, rate=self.drop_prop)
        else:
           mu_out = mu_in  
        return mu_out
    
class BaSIS_Batch_Normalization(keras.layers.Layer):
    def __init__(self, var_epsilon):
        super(BaSIS_Batch_Normalization, self).__init__()
        self.var_epsilon = var_epsilon

    def call(self, mu_in):
        mean, variance = tf.nn.moments(mu_in, [0, 1, 2])
        mu_out = tf.nn.batch_normalization(mu_in, mean, variance, offset=None, scale=None,
                                           variance_epsilon=self.var_epsilon)
        return mu_out

            
class Density_prop_CNN_MNIST(tf.keras.Model):
  def __init__(self, tau, kernel_size,num_kernel, pooling_size, pooling_stride, pooling_pad, units, name=None):
    super(Density_prop_CNN_MNIST, self).__init__()
    self.tau = tau
    self.kernel_size = kernel_size
    self.num_kernel = num_kernel
    self.pooling_size = pooling_size
    self.pooling_stride = pooling_stride
    self.pooling_pad = pooling_pad
    self.units = units
    self.conv_1 = BaSIS_first_Conv(tau = self.tau, kernel_size=self.kernel_size[0], kernel_num=self.num_kernel[0], kernel_stride=1, padding="VALID")
    self.relu_1 = BaSIS_ReLU()
    self.maxpooling_1 = BaSIS_MaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0], pooling_pad=self.pooling_pad)
    self.fc_1 = BaSIS_Flatten_and_FC(self.units, tau = self.tau)   
    self.mysoftma = softmax()
    
    
  def call(self, inputs, training=True):
    mu1 = self.conv_1(inputs) 
    mu2 = self.relu_1(mu1) 
    mu3 = self.maxpooling_1(mu2) 
    mu4 = self.fc_1(mu3)   
    mu_out = self.mysoftma(mu4)    
    return mu_out, mu4


class Density_prop_CNN_CIFAR(tf.keras.Model):
  def __init__(self, kernel_size,n_kernels,regularization,pooling_size, pooling_stride, pooling_pad,n_labels,  drop_prop=0.2,var_epsilon=1e-4,name=None):
    super(Density_prop_CNN_CIFAR, self).__init__()

    self.num_kernel = n_kernels
    self.n_out = n_labels
    self.pooling_size = pooling_size
    self.pooling_stride = pooling_stride
    self.pooling_pad = pooling_pad
    self.kernel_size = kernel_size
    self.r = regularization
    self.var_epsilon = var_epsilon
    self.drop_prop = drop_prop

    self.conv_1 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[0], kernel_num=self.num_kernel[0], padding='VALID')
    self.conv_2 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[1], kernel_num=self.num_kernel[1],   padding='SAME')
    self.conv_3 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[2], kernel_num=self.num_kernel[2],   padding='SAME')
    self.conv_4 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[3], kernel_num=self.num_kernel[3],   padding='SAME')
    self.conv_5 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[4], kernel_num=self.num_kernel[4],   padding='SAME')
    self.conv_6 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[5], kernel_num=self.num_kernel[5],  padding='SAME')
    self.conv_7 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[6], kernel_num=self.num_kernel[6],   padding='SAME')
    self.conv_8 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[7], kernel_num=self.num_kernel[7],   padding='SAME')
    self.conv_9 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[8], kernel_num=self.num_kernel[8],   padding='SAME')
    self.conv_10 = BaSIS_Conv(tau = self.r,kernel_size=self.kernel_size[9], kernel_num=self.num_kernel[9],   padding='SAME')

    self.fc_1 = BaSIS_Flatten_and_FC(self.n_out, tau = self.r)   
    
    self.elu_1 = BaSIS_eLU()
    self.maxpooling_1 = BaSIS_MaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0], pooling_pad=self.pooling_pad)
    self.maxpooling_2 = BaSIS_MaxPooling(pooling_size=self.pooling_size[1], pooling_stride=self.pooling_stride[1],   pooling_pad=self.pooling_pad)
    self.maxpooling_3 = BaSIS_MaxPooling(pooling_size=self.pooling_size[2], pooling_stride=self.pooling_stride[2],   pooling_pad=self.pooling_pad)
    self.maxpooling_4 = BaSIS_MaxPooling(pooling_size=self.pooling_size[3], pooling_stride=self.pooling_stride[3],   pooling_pad=self.pooling_pad)
    self.maxpooling_5 = BaSIS_MaxPooling(pooling_size=self.pooling_size[4], pooling_stride=self.pooling_stride[4],   pooling_pad=self.pooling_pad)
    self.dropout_1 = BaSIS_Dropout(self.drop_prop)        
    self.batch_norm = BaSIS_Batch_Normalization(self.var_epsilon)
    self.mysoftma = softmax()
  def call(self, inputs, training=True):
    mu = self.conv_1(inputs)    # [28,28,32]   
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.maxpooling_1(mu) #[14,14,32]

    mu = self.conv_2(mu)      #[14,14,32]  
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.conv_3(mu)     #[14,14,32]   
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.maxpooling_2(mu) #[7,7,32]
    mu = self.dropout_1(mu, Training=training)  

    mu = self.conv_4(mu)       #[7,7,32]
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.conv_5(mu)      #[7,7,64] 
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.maxpooling_3(mu) 
    mu = self.dropout_1(mu, Training=training)
    
    mu = self.conv_6(mu)      
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.conv_7(mu)       
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.maxpooling_4(mu) 
    mu = self.dropout_1(mu, Training=training)

    mu = self.conv_8(mu)      
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.conv_9(mu)       
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.maxpooling_5(mu) 
    mu = self.dropout_1(mu, Training=training)
    
    mu = self.conv_10(mu)      
    mu = self.elu_1(mu)
    mu = self.batch_norm(mu)
    mu = self.fc_1(mu) 
    mu_out = self.mysoftma(mu)
    return mu_out, mu
  		
def ce_particles(y_out,y_label,N, weights):
    batch = y_out.shape[0]
    
    num_labels = y_out.shape[-1]
    y = tf.cast(y_label, tf.int32)
    yy = tf.expand_dims(y,axis=1)
    y = tf.broadcast_to(yy, [ batch, N])
    ce = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_out, labels = y),axis=0)  #shape=[N, 1] 
    #RESAMPLE 
    like, indeces =resample_like(ce,N, weights) # indices shape = [N,1]
    return like, indeces

		
def ce_particles_with_L2(part, y_out,y_label,N, weights, last = False):
    batch = y_out.shape[0]
    y = tf.cast(y_label, tf.int32)
    yy = tf.expand_dims(y,axis=1)
    y = tf.broadcast_to(yy, [ batch, N])
    if last:
        L2 = tf.reduce_sum(tf.square(part), axis = [0,1,2,3])
        ce = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_out, labels = y),axis=0) + L2  #shape=[N, 1] 
    else:
        L2 = tf.reduce_sum(tf.square(part), axis = [1,2])
        ce = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_out, labels = y),axis=0) #+ L2  #shape=[N, 1] 
    
    #RESAMPLE 
    like, indeces =resample_like(ce,N, weights) # indices shape = [N,1]
    return like, indeces


def indeces(weights,N):
  ind = tf.random.categorical(weights,N)
  return ind

def resample_like(ce,N, weights):
    if weights == 'inv':
        ratios = tf.math.reciprocal(ce)
    else:
        ratios = tf.math.exp(-ce)
    u=ratios/tf.math.reduce_sum(ratios)
    u=tf.reshape(u, [1,N])
    ind=indeces(u,N)
    ind=tf.reshape(ind,shape=[N,1])
    u=(1/N)*np.ones(N)
    return u, ind

def batch_norm(mu_in, var_epsilon=1e-4):
    mean, variance = tf.nn.moments(mu_in, [0, 1, 2])
    mu_out = tf.nn.batch_normalization(mu_in, mean, variance, offset=None, scale=None,
                                           variance_epsilon=var_epsilon)
    return mu_out


def max_pool_2x2(mu_in, pooling_size=2, pooling_stride=2):
        mu_out, argmax_out = tf.nn.max_pool_with_argmax(mu_in, ksize=[1, pooling_size, pooling_size, 1], strides=[1, pooling_stride, pooling_stride, 1], padding= "SAME", data_format='NHWC') #shape=[batch_zise, new_size,new_size,num_channel]
        return mu_out

def BaSIS_Conv_likelihood_MNIST( x, W1, W2,  y_label, kernel_size=5, kernel_num=32, kernel_stride=1,init_std=0.1, N= 1000, sigma_part = 0.1, padding="VALID", weights = 'inv'):   
        batch = x.shape[0]
        num_channel = x.shape[-1]
        y_label =tf.cast(y_label,tf.int32)
        
        noise = tf.random.normal([kernel_size,kernel_size, num_channel, kernel_num, N], stddev = sigma_part) # creating noise 
        part = tf.math.add(noise ,W1) 
        part_reshape = tf.reshape(part, [kernel_size,kernel_size, num_channel, -1])
        z = tf.nn.conv2d(x, part_reshape,  strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')
        # shape after conv=[batch_size, image_size,image_size,kernel_num*N]
        g = tf.nn.relu(z)
        p = max_pool_2x2(g)
        part_out = tf.reshape(p,[batch, p.shape[1], p.shape[2], kernel_num, N])
        part_out = tf.transpose(part_out, [0,4, 1, 2, 3]) #output = [ batch,N, new_size, new_size, Kernel_num]
        new_size = part_out.shape[2]
        b = tf.reshape(part_out, [batch, part_out.shape[1], new_size*new_size* kernel_num]) #shape=[batch_size, new_size*new_size* kernel_num]   
        fc1 = tf.matmul(b, W2) #shape=[batch_size,N, num_labels]      
        y_out = tf.nn.softmax(fc1) #shape=[batch_size,N,  num_labels]  
        like_1, indeces = ce_particles(fc1, y_label,N, weights)
        part_conv_layer= tf.gather_nd(tf.transpose(part, [4,0, 1, 2, 3]) , indeces, batch_dims=0) #shape = [N,kernel_size,kernel_size, num_channel, kernel_num]
        part_conv_layer = tf.transpose(part_conv_layer , [1,2,3,4,0])
        return part_conv_layer
                        
def BaSIS_Flatten_and_FC_MNIST(x,W1, W2,  y_label, kernel_size=5, kernel_num=32,units = 10, kernel_stride=1,init_std=0.1, N= 1000, sigma_part = 0.1, padding="VALID", weights = 'inv'):   
        
        batch = x.shape[0] #shape=[batch_size, im_size, im_size, num_channel] 
        
        z = tf.nn.conv2d(x, W1, strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g = tf.nn.relu(z)
        p = max_pool_2x2(g)
        new_size = p.shape[1]
        b = tf.reshape(p, [batch, -1]) #shape=[batch_size, new_size*new_size* num_filters] 
        
        noise = tf.random.normal([N,new_size*new_size*kernel_num,units], stddev = sigma_part) # creating particles 
        part = tf.math.add(noise , W2 )#shape = [N, new_size*new_size*num_filters, num_labels]
        fc1 = tf.matmul(b, part) 
        fc2 = tf.reshape(fc1, [batch,N,units])
        y_out = tf.nn.softmax(fc1)
        y_out = tf.reshape(y_out, [batch,N,units])
        like_2, indeces = ce_particles(fc2, y_label, N, weights)
        part_fc_layer= tf.gather_nd(part , indeces, batch_dims=0) #shape = [N,patch_size,patch_size, num_channel, num_filter[0]]
        return part_fc_layer,  y_out  

def CNN_model_test_MNIST(x, W1, W2,N , kernel_stride=1, padding ="VALID"):
   
        x = tf.cast(x ,dtype =tf.float32)
        W1 = tf.cast(W1 ,dtype =tf.float32)
        W2 = tf.cast(W2 ,dtype =tf.float32)
        
        batch = x.shape[0]
        num_labels = 10
        patch_size = W1.shape[1]
        num_filters = W1.shape[3]
        num_channel = x.shape[-1]
        part_reshape = tf.reshape(W1, [patch_size,patch_size, num_channel,num_filters*N ])
        z = tf.nn.conv2d(x, part_reshape, strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g = tf.nn.relu(z)
        part_out = tf.reshape(g,[batch, z.shape[1], z.shape[1], num_filters, N])
        mu_g = tf.math.reduce_mean(part_out, axis =4)
        mu_g2= tf.reshape(mu_g,[batch, z.shape[1], z.shape[1], num_filters])
        p = max_pool_2x2(mu_g2)
        new_size = p.shape[2]
        b = tf.reshape(p, [batch, new_size*new_size*num_filters]) #shape=[batch_size, new_size*new_size* num_filters[0]]  
        
        W2_reshape = tf.reshape(W2, [N,new_size*new_size*num_filters, num_labels])
        fc1 = tf.matmul(b, W2_reshape)  #shape=[N, batch_size, num_labels]      
        y_out_part = tf.nn.softmax(fc1 ) #shape=[N, batch_size, num_labels]  
        y_out = tf.math.reduce_mean(y_out_part, axis = 0)
        return y_out_part, y_out  , fc1


def conv_particles(x, W1,kernel_size, kernel_stride, num_channel, kernel_num,  N ,sigma_part, padding = "SAME"):
    noise = tf.random.normal([kernel_size,kernel_size, num_channel, kernel_num, N], stddev = sigma_part) # creating particles 
    part = tf.math.add(noise ,W1) #shape = [kernel_size,kernel_size, num_channel, num_filter[0], N]
    part_reshape = tf.reshape(part, [kernel_size,kernel_size, num_channel, -1])
    z = tf.nn.conv2d(x, part_reshape,  strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')
    #output = [ batch, new_size, new_size, Kernel_num*N]
    return z, part
def reshape_particles(z, kernel_num, N):
    image_size1=z.shape[1]
    z = tf.reshape(z,[z.shape[0], image_size1, image_size1, kernel_num, N])
    z = tf.transpose(z, [0,4, 1, 2, 3])
    z = tf.reshape(z,[-1, image_size1, image_size1, kernel_num] )
    return z

def BaSIS_Conv_likelihood_CIFAR(i, x, W1, W2,W3, W4, W5, W6, W7,W8, W9,W10, W11,    y_label, kernel_num, kernel_size, kernel_stride=1, n_labels = 3,drop_prop = 0.2, init_std=0.1, N= 100, sigma_part = 0.01, padding="SAME", weights = 'inv'):   
        batch = x.shape[0]
        num_channel = x.shape[-1]
        y_label =tf.cast(y_label,tf.int32)
        

        if i ==1:                       
            z, part = conv_particles(x,  W1,kernel_size[0], kernel_stride, num_channel, kernel_num[0],  N,  sigma_part, padding='VALID')
            # shape after conv=[batch_size, image_size,image_size,kernel_num*N]
        else:
            z = tf.nn.conv2d(x, W1,  strides=[1,kernel_stride,kernel_stride,1], padding='VALID', data_format='NHWC')
        g = tf.nn.elu(z)
        b = batch_norm(g)
        if i ==1:  
            b = reshape_particles(b,kernel_num[0], N)
        p = max_pool_2x2(b)
        if i ==2:   
            z2 , part= conv_particles(p, W2,kernel_size[1], kernel_stride, kernel_num[0], kernel_num[1], N , sigma_part, padding)
            # shape after conv=[batch_size, image_size,image_size,kernel_num*N]
        else:
            z2 = tf.nn.conv2d(p, W2,  strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')
            # shape=[batch,N, image_size,image_size,num_filters[1]]
        g2 = tf.nn.elu(z2)
        b2 = batch_norm(g2)
        if i ==2:  
            b2 = reshape_particles(b2,kernel_num[1], N)
        if i ==3:   
            z3 , part= conv_particles(b2, W3,kernel_size[2], kernel_stride, kernel_num[1], kernel_num[2], N ,sigma_part, padding)
            # shape after conv=[batch_size, image_size,image_size,kernel_num*N]
        else:
            z3 = tf.nn.conv2d(b2, W3,  strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g3 = tf.nn.elu(z3)
        b3 = batch_norm(g3) 
        if i ==3:  
            b3 = reshape_particles(b3,kernel_num[2], N)
        p2 = max_pool_2x2(b3)
        p2 = tf.nn.dropout(p2, rate=drop_prop)

        if i ==4:   
            z4, part = conv_particles(p2, W4,kernel_size[3], kernel_stride, kernel_num[2], kernel_num[3], N ,sigma_part, padding= 'SAME')
            # shape after conv=[batch_size, image_size,image_size,kernel_num*N]
        else:
            z4 = tf.nn.conv2d(p2, W4,  strides=[1,kernel_stride,kernel_stride,1], padding ='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g4 = tf.nn.elu(z4)
        b4 = batch_norm(g4) 
        if i ==4:  
            b4 = reshape_particles(b4,kernel_num[3], N)
        if i ==5:   
            z5, part = conv_particles(b4, W5,kernel_size[4], kernel_stride,kernel_num[3], kernel_num[4],  N ,sigma_part,padding='SAME')
            # shape after conv=[batch_size,N, image_size,image_size,kernel_num]
        else:
            z5 = tf.nn.conv2d(b4, W5,  strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g5 = tf.nn.elu(z5)
        b5 = batch_norm(g5) 
        if i ==5:  
            b5 = reshape_particles(b5,kernel_num[4], N)
        p5 = max_pool_2x2(b5)
        p5 = tf.nn.dropout(p5, rate=drop_prop)

        if i ==6:   
            z6 , part= conv_particles(p5, W6,kernel_size[5], kernel_stride, kernel_num[4], kernel_num[5], N ,sigma_part, padding= 'SAME')
            # shape after conv=[batch_size,N,  image_size,image_size,kernel_num]
        else:
            z6 = tf.nn.conv2d(p5, W6,  strides=[1,kernel_stride,kernel_stride,1], padding ='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g6 = tf.nn.elu(z6) 
        b6 = batch_norm(g6) 
        if i ==6:  
            b6 = reshape_particles(b6,kernel_num[5], N)
        if i ==7:   
            z7, part = conv_particles(b6, W7,kernel_size[6], kernel_stride,kernel_num[5], kernel_num[6],  N ,sigma_part,padding='SAME')
            # shape after conv=[batch_size,N, image_size,image_size,kernel_num]
        else:
            z7 = tf.nn.conv2d(b6, W7,  strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g7 = tf.nn.elu(z7)
        b7 = batch_norm(g7) 
        if i ==7:  
            b7 = reshape_particles(b7,kernel_num[6], N)
        p7 = max_pool_2x2(b7)
        p7 = tf.nn.dropout(p7, rate=drop_prop)

        if i ==8:   
            z8 , part= conv_particles(p7, W8,kernel_size[7], kernel_stride, kernel_num[6], kernel_num[7], N ,sigma_part, padding= 'SAME')
            # shape after conv=[batch_size,N,  image_size,image_size,kernel_num]
        else:
            z8 = tf.nn.conv2d(p7, W8,  strides=[1,kernel_stride,kernel_stride,1], padding ='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g8 = tf.nn.elu(z8) 
        b8 = batch_norm(g8) 
        if i ==8:  
            b8 = reshape_particles(b8,kernel_num[7], N)
        if i ==9:   
            z9, part = conv_particles(b8, W9,kernel_size[8], kernel_stride,kernel_num[7], kernel_num[8],  N ,sigma_part,padding='SAME')
            # shape after conv=[batch_size,N, image_size,image_size,kernel_num]
        else:
            z9 = tf.nn.conv2d(b8, W9,  strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g9 = tf.nn.elu(z9)
        b9 = batch_norm(g9) 
        if i ==9:  
            b9 = reshape_particles(b9,kernel_num[8], N)
        p9 = max_pool_2x2(b9)
        p9 = tf.nn.dropout(p9, rate=drop_prop)

        if i ==10:   
            z10 , part= conv_particles(p9, W10,kernel_size[9], kernel_stride, kernel_num[8], kernel_num[9], N ,sigma_part, padding= 'SAME')
            # shape after conv=[batch_size,N,  image_size,image_size,kernel_num]
        else:
            z10 = tf.nn.conv2d(p9, W10,  strides=[1,kernel_stride,kernel_stride,1], padding ='SAME', data_format='NHWC')# shape=[batch,N, image_size,image_size,num_filters[1]]
        g10 = tf.nn.elu(z10) 
        b10 = batch_norm(g10) 
        if i ==10:  
            b10 = reshape_particles(b10,kernel_num[9], N)
        
        new_size = b10.shape[2]
        
        b = tf.reshape(b10, [batch, N, new_size*new_size* kernel_num[9]])
        fc1 = tf.matmul(b, W11) #shape=[batch_size,N, num_labels]      
        y_out = tf.nn.softmax(fc1) #shape=[batch_size,N,  num_labels]  
        like_1, indeces = ce_particles(fc1, y_label,N, weights)
        part_conv_layer= tf.gather_nd(tf.transpose(part, [4,0, 1, 2, 3]) , indeces, batch_dims=0) #shape = [N,kernel_size,kernel_size, num_channel, kernel_num]
        part_conv_layer = tf.transpose(part_conv_layer , [1,2,3,4,0])
        return part_conv_layer
                                                    
def BaSIS_Flatten_and_FC_CIFAR(x,W1, W2,W3,W4,W5,W6,W7,W8, W9,W10, W11,  y_label,units = 10, kernel_stride=1,init_std=0.1, N= 1000, sigma_part = 0.1, drop_prop=0.2,padding="SAME", weights = 'inv'):   
        
        batch = x.shape[0] #shape=[batch_size, im_size, im_size, num_channel] 
        
        z = tf.nn.conv2d(x, W1, strides=[1,kernel_stride,kernel_stride,1], padding='VALID', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g = tf.nn.elu(z)
        b = batch_norm(g)
        p = max_pool_2x2(b)

        z2 = tf.nn.conv2d(p, W2, strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g2 = tf.nn.elu(z2)
        b2 = batch_norm(g2)
        z3 = tf.nn.conv2d(b2, W3, strides=[1,kernel_stride,kernel_stride,1], padding=padding, data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g3 = tf.nn.elu(z3)
        b3 = batch_norm(g3)
        p3 = max_pool_2x2(b3)
        p3 = tf.nn.dropout(p3, rate=drop_prop)

        z4 = tf.nn.conv2d(p3, W4, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g4 = tf.nn.elu(z4)
        b4 = batch_norm(g4)
        z5 = tf.nn.conv2d(b4, W5, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g5 = tf.nn.elu(z5)
        b5 = batch_norm(g5)
        p5 = max_pool_2x2(b5)
        p5 = tf.nn.dropout(p5, rate=drop_prop)

        z6 = tf.nn.conv2d(p5, W6, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g6 = tf.nn.elu(z6)
        b6 = batch_norm(g6)
        z7 = tf.nn.conv2d(b6, W7, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g7 = tf.nn.elu(z7)
        b7 = batch_norm(g7)
        p7 = max_pool_2x2(b7)
        p7 = tf.nn.dropout(p7, rate=drop_prop)

        z8 = tf.nn.conv2d(p7, W8, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g8 = tf.nn.elu(z8)
        b8 = batch_norm(g8)
        z9 = tf.nn.conv2d(b8, W9, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g9 = tf.nn.elu(z9)
        b9 = batch_norm(g9)
        p9 = max_pool_2x2(b9)
        p9 = tf.nn.dropout(p9, rate=drop_prop)

        z10 = tf.nn.conv2d(p9, W10, strides=[1,kernel_stride,kernel_stride,1], padding='SAME', data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters]
        g10 = tf.nn.elu(z10)
        b10 = batch_norm(g10)
        
        new_size = b10.shape[1]
        b = tf.reshape(b10, [batch, -1]) #shape=[batch_size, new_size*new_size* num_filters] 
        kernel_num_fin = b10.shape[-1]
        
        noise = tf.random.normal([N,new_size*new_size*kernel_num_fin,units], stddev = sigma_part) # creating particles 
        part = tf.math.add(noise , W11 )#shape = [N, new_size*new_size*num_filters, num_labels]
        fc1 = tf.matmul(b, part) 
        fc2 = tf.reshape(fc1, [batch,N,units])
        y_out = tf.nn.softmax(fc1)
        y_out = tf.reshape(y_out, [batch,N,units])
        like_2, indeces = ce_particles(fc2, y_label, N, weights)
        part_fc_layer= tf.gather_nd(part , indeces, batch_dims=0) #shape = [N,patch_size,patch_size, num_channel, num_filter[0]]
        return part_fc_layer,  y_out  


def CNN_model_test_CIFAR(x, W1, W2,W3, W4, W5, W6, W7,W8,W9,W10,W11, num_filters, kernel_size , N =100,  kernel_stride=1,num_labels = 10, padding ="SAME"):
   
        x = tf.cast(x ,dtype =tf.float32)
        W1 = tf.cast(W1 ,dtype =tf.float32)
        W2 = tf.cast(W2 ,dtype =tf.float32)
        W3 = tf.cast(W3 ,dtype =tf.float32)
        W4 = tf.cast(W4 ,dtype =tf.float32)
        W5 = tf.cast(W5 ,dtype =tf.float32)
        W6 = tf.cast(W6 ,dtype =tf.float32)
        W7 = tf.cast(W7 ,dtype =tf.float32)
        W8 = tf.cast(W8 ,dtype =tf.float32)
        W9 = tf.cast(W9 ,dtype =tf.float32)
        W10 = tf.cast(W10 ,dtype =tf.float32)
        W11 = tf.cast(W11 ,dtype =tf.float32)

        batch = x.shape[0]
        patch_size = kernel_size
       
        num_channel = x.shape[-1]
      
        part_reshape = tf.reshape(W1, [patch_size[0],patch_size[0], num_channel,num_filters[0]*N ])
        
        z = tf.nn.conv2d(x, part_reshape, strides=[1,kernel_stride,kernel_stride,1], padding="VALID",data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g = tf.nn.elu(z)
        b = batch_norm(g)
        p = max_pool_2x2(b)
        part_out = tf.reshape(p,[batch, p.shape[1], p.shape[1], num_filters[0], N])
        mu_g = tf.math.reduce_mean(part_out, axis =4)
        mu_g= tf.squeeze(mu_g)

        part_reshape2 = tf.reshape(W2, [patch_size[1],patch_size[1], num_filters[0],num_filters[1]*N ])
        z2 = tf.nn.conv2d(mu_g, part_reshape2, strides=[1,kernel_stride,kernel_stride,1], padding=padding,data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g2 = tf.nn.elu(z2)
        b2 = batch_norm(g2)
        part_out2 = tf.reshape(b2,[batch, b2.shape[1], b2.shape[1], num_filters[1], N])
        mu_p = tf.math.reduce_mean(part_out2, axis =4)
        mu_p = tf.squeeze(mu_p)
        
        part_reshape3 = tf.reshape(W3, [patch_size[2],patch_size[2], num_filters[1],num_filters[2]*N ])
        z3 = tf.nn.conv2d(mu_p, part_reshape3, strides=[1,kernel_stride,kernel_stride,1], padding=padding,data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g3 = tf.nn.elu(z3)
        b3 = batch_norm(g3)
        p3 = max_pool_2x2(b3)
        part_out3 = tf.reshape(p3,[batch, p3.shape[1], p3.shape[1], num_filters[2], N])
        mu_g3 = tf.math.reduce_mean(part_out3, axis =4)
        mu_g3 = tf.squeeze(mu_g3)
       
        part_reshape4 = tf.reshape(W4, [patch_size[3],patch_size[3], num_filters[2],num_filters[3]*N ])
        z4 = tf.nn.conv2d(mu_g3, part_reshape4, strides=[1,kernel_stride,kernel_stride,1], padding="SAME",data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g4 = tf.nn.elu(z4)
        b4 = batch_norm(g4)
        part_out4 = tf.reshape(b4,[batch, b4.shape[1], b4.shape[1], num_filters[3], N])
        mu_p2 = tf.math.reduce_mean(part_out4, axis =4)
        mu_p2 = tf.squeeze(mu_p2)
        
        part_reshape5 = tf.reshape(W5, [patch_size[4],patch_size[4], num_filters[3],num_filters[4]*N ])
        z5 = tf.nn.conv2d(mu_p2, part_reshape5, strides=[1,kernel_stride,kernel_stride,1],padding="SAME", data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g5 = tf.nn.elu(z5)
        b5 = batch_norm(g5)
        p5 = max_pool_2x2(b5)
        part_out5 = tf.reshape(p5,[batch, p5.shape[1], p5.shape[1], num_filters[4], N])
        mu_g5 = tf.math.reduce_mean(part_out5, axis =4)
        mu_g5 = tf.squeeze(mu_g5)

        
        part_reshape6 = tf.reshape(W6, [patch_size[5],patch_size[5], num_filters[4],num_filters[5]*N ])
        z6 = tf.nn.conv2d(mu_g5, part_reshape6, strides=[1,kernel_stride,kernel_stride,1],padding="SAME", data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g6 = tf.nn.elu(z6)
        b6 = batch_norm(g6)
        part_out6 = tf.reshape(b6,[batch, b6.shape[1], b6.shape[1], num_filters[5], N])
        mu_p6 = tf.math.reduce_mean(part_out6, axis =4)
        mu_p6 = tf.squeeze(mu_p6)

        part_reshape7 = tf.reshape(W7, [patch_size[6],patch_size[6], num_filters[5],num_filters[6]*N ])
        z7 = tf.nn.conv2d(mu_p6, part_reshape7, strides=[1,kernel_stride,kernel_stride,1],padding="SAME", data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g7 = tf.nn.elu(z7)
        b7 = batch_norm(g7)
        p7 = max_pool_2x2(b7)
        part_out7 = tf.reshape(p7,[batch, p7.shape[1], p7.shape[1], num_filters[6], N])
        mu_g7 = tf.math.reduce_mean(part_out7, axis =4)
        mu_g7 = tf.squeeze(mu_g7)

        part_reshape8 = tf.reshape(W8, [patch_size[7],patch_size[7], num_filters[6],num_filters[7]*N ])
        z8 = tf.nn.conv2d(mu_g7, part_reshape8, strides=[1,kernel_stride,kernel_stride,1],padding="SAME", data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g8 = tf.nn.elu(z8)
        b8 = batch_norm(g8)
        part_out8 = tf.reshape(b8,[batch, b8.shape[1], b8.shape[1], num_filters[7], N])
        mu_p8 = tf.math.reduce_mean(part_out8, axis =4)
        mu_p8 = tf.squeeze(mu_p8)

        part_reshape9 = tf.reshape(W9, [patch_size[8],patch_size[8], num_filters[7],num_filters[8]*N ])
        z9 = tf.nn.conv2d(mu_p8, part_reshape9, strides=[1,kernel_stride,kernel_stride,1],padding="SAME", data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g9 = tf.nn.elu(z9)
        b9 = batch_norm(g9)
        p9 = max_pool_2x2(b9)
        part_out9 = tf.reshape(p9,[batch, p9.shape[1], p9.shape[1], num_filters[8], N])
        mu_g9 = tf.math.reduce_mean(part_out9, axis =4)
    
        part_reshape10 = tf.reshape(W10, [patch_size[9],patch_size[9], num_filters[8],num_filters[9]*N ])
        z10 = tf.nn.conv2d(mu_g9, part_reshape10, strides=[1,kernel_stride,kernel_stride,1],padding="SAME", data_format='NHWC')# shape=[batch_size, image_size,image_size,num_filters[0]*N]
        g10 = tf.nn.elu(z10)
        b10 = batch_norm(g10)
        part_out10 = tf.reshape(b10,[batch, b10.shape[1], b10.shape[1], num_filters[9], N])
        mu_p10 = tf.math.reduce_mean(part_out10, axis =4)
    
        new_size = mu_p10.shape[2]
        b = tf.reshape(mu_p10, [batch, new_size*new_size*num_filters[9]]) #shape=[batch_size, new_size*new_size* num_filters[0]]  
        
        W11_reshape = tf.reshape(W11, [N,new_size*new_size*num_filters[9], num_labels])
        fc1 = tf.matmul(b, W11_reshape)  #shape=[N, batch_size, num_labels]     
        y_out_part = tf.nn.softmax(fc1 ) #shape=[N, batch_size, num_labels]  
        y_out = tf.math.reduce_mean(y_out_part, axis = 0)
        return y_out_part, y_out  , fc1

