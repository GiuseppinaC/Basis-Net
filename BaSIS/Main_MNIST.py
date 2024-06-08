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
import  sys
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
from model_functions import *
from other_functions import *
import argparse

# Initialize argparse
parser = argparse.ArgumentParser(description='This script is used to train & test the BaSIS-Net model on the MNIST dataset. Specify the path to save the model, and other hyperparameters.')

# Add arguments
parser = argparse.ArgumentParser(description="Hyperparameters for the model")

parser.add_argument('--folder', type=str, required=True, 
                        help='Folder to save model and logs')
parser.add_argument('--input_dim', type=int,  default=28, 
                        help='size of the input - 28 for MNIST')
parser.add_argument('--num_kernels', type=int, nargs='+', default=[32], 
                        help='Number of kernels for convolutional layers')
parser.add_argument('--kernels_size', type=int, nargs='+', default=[5], 
                        help='Kernel size for convolutional layers')
parser.add_argument('--maxpooling_size', type=int, default=[2], 
                        help='Max pooling size')
parser.add_argument('--maxpooling_stride', type=int, default=[2], 
                        help='Stride for max pooling')
parser.add_argument('--maxpooling_pad', type=str, default='SAME', 
                        help='Padding for max pooling')
parser.add_argument('--class_num', type=int, default=10, 
                        help='Number of classes')
parser.add_argument('--batch_size', type=int, default=50, 
                        help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate')
parser.add_argument('--lr_end', type=float, default=0.0001, 
                        help='Ending learning rate')
parser.add_argument('--reg_factor', type=float, default=0.003, 
                        help='Regularization factor')
parser.add_argument('--N', type=int, default=100, 
                        help='Number of particles')
parser.add_argument('--sigma_particles_conv', type=float, default=0.0001, 
                        help='Sigma for particles in convolutional layers')
parser.add_argument('--sigma_particles_fc', type=float, default=0.0001, 
                        help='Sigma for particles in fully connected layers')
parser.add_argument('--init_sigma', type=float, default=0.0001, 
                        help='Initial sigma for particles')
parser.add_argument('--Training', type=bool, default=False, 
                        help='Flag to indicate if training should be performed')
parser.add_argument('--continue_training', type=bool, default=False, 
                        help='Flag to indicate if training should continue from a checkpoint')
parser.add_argument('--saved_model_epochs', type=int, default=0, 
                        help='Number of epochs of saved model - to continue training')
parser.add_argument('--Testing', type=bool, default=False, 
                        help='Flag to indicate if testing should be performed')
parser.add_argument('--weights', type=str, default='exp', 
                        help='current implementation is for exp - other option inv')
parser.add_argument('--Random_noise', type=bool, default=False, 
                        help='Flag to indicate if random noise should be added to testing dataset')
parser.add_argument('--gaussian_noise_var', type=float, default=0.01, 
                        help='Variance of the Gaussian noise to be added')
parser.add_argument('--Adversarial_noise', type=bool, default=False, 
                        help='Flag to indicate if adversarial noise should be added to testing dataset')
parser.add_argument('--epsilon', type=float, default=0, 
                        help='Epsilon value for adversarial noise')
parser.add_argument('--adversary_target_cls', type=int, default=3, 
                        help='Target class for adversarial noise')
parser.add_argument('--Targeted', type=bool, default=False, 
                        help='Flag to indicate if the adversarial attack is targeted')
parser.add_argument('--histogram', type=bool, default=False, 
                        help='Flag to indicate if histograms should be generated')
# Parse arguments
args = parser.parse_args()

# Accessing all arguments
folder = args.folder
input_dim = args.input_dim
num_kernels = args.num_kernels
kernels_size = args.kernels_size
maxpooling_size = args.maxpooling_size
maxpooling_stride = args.maxpooling_stride
maxpooling_pad = args.maxpooling_pad
class_num = args.class_num
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
lr_end = args.lr_end
reg_factor = args.reg_factor
N = args.N
sigma_particles_conv = args.sigma_particles_conv
sigma_particles_fc = args.sigma_particles_fc
init_sigma = args.init_sigma
Training = args.Training
continue_training = args.continue_training
saved_model_epochs = args.saved_model_epochs
Testing = args.Testing
weights = args.weights
Random_noise = args.Random_noise
gaussian_noise_var = args.gaussian_noise_var
Adversarial_noise = args.Adversarial_noise
epsilon = args.epsilon
adversary_target_cls = args.adversary_target_cls
Targeted = args.Targeted
histogram = args.histogram
    
# Printing all arguments
print("Arguments:")
print(f"Folder: {folder}")
print(f"Number of Kernels: {num_kernels}")
print(f"Kernel Size: {kernels_size}")
print(f"Max Pooling Size: {maxpooling_size}")
print(f"Max Pooling Stride: {maxpooling_stride}")
print(f"Max Pooling Pad: {maxpooling_pad}")
print(f"Number of Classes: {class_num}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Learning Rate: {lr}")
print(f"Ending Learning Rate: {lr_end}")
print(f"Regularization Factor: {reg_factor}")
print(f"Number of Particles: {N}")
print(f"Sigma for Particles in Conv Layers: {sigma_particles_conv}")
print(f"Sigma for Particles in FC Layers: {sigma_particles_fc}")
print(f"Initial Sigma: {init_sigma}")
print(f"Training: {Training}")
print(f"Continue Training: {continue_training}")
print(f"Saved Model Epochs: {saved_model_epochs}")
print(f"Testing: {Testing}")
print(f"Weights: {weights}")
print(f"Random Noise: {Random_noise}")
print(f"Gaussian Noise Variance: {gaussian_noise_var}")
print(f"Adversarial Noise: {Adversarial_noise}")
print(f"Epsilon: {epsilon}")
print(f"Adversary Target Class: {adversary_target_cls}")
print(f"Targeted: {Targeted}")
print(f"Histogram: {histogram}")


def main_function(folder, input_dim, num_kernels, kernels_size, maxpooling_size, maxpooling_stride, maxpooling_pad, class_num, batch_size,
        epochs, lr, lr_end, reg_factor, N,sigma_particles_conv,sigma_particles_fc, init_sigma, 
        Training , continue_training ,  saved_model_epochs, Testing , weights,
        Random_noise, gaussian_noise_var, Adversarial_noise, epsilon, adversary_target_cls, Targeted,
         histogram):   

    if weights == 'inv'   :
        PATH = folder+'/saved_models/epoch_{}/{}_particles/inv_ce/ '.format( epochs,N) 
    else:   
        PATH = folder+'/saved_models/epoch_{}/{}_particles/exp_ce/'.format( epochs,N) 
     
    if not os.path.exists(PATH):
        os.makedirs(PATH)
        print('created folder')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
    cnn_model = Density_prop_CNN_MNIST(tau = reg_factor,kernel_size=kernels_size,num_kernel=num_kernels, pooling_size=maxpooling_size, pooling_stride=maxpooling_stride, pooling_pad=maxpooling_pad, units=class_num, name = 'vdp_cnn')       
    num_train_steps = epochs * int(x_train.shape[0] /batch_size)

    
    @tf.function  
    def get_gradient(x, y):
        with tf.GradientTape() as tape:
            mu_out, fc = cnn_model(x, training=True)  
            cnn_model.trainable = True         
            loss_final = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = fc, labels = y),axis=0) 
            regularization_loss=tf.math.add_n(cnn_model.losses)             
            loss = loss_final + regularization_loss           
        gradients = tape.gradient(loss, cnn_model.trainable_weights)  
               
        gradients = [(tf.where(tf.math.is_nan(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients = [(tf.where(tf.math.is_inf(grad), tf.constant(1.0e-5, shape=grad.shape), grad)) for grad in gradients]
        gradients1 = gradients[0]
        gradients2 = gradients[1]
        grad1 = tf.convert_to_tensor(gradients1)
        grad2 = tf.convert_to_tensor(gradients2)
        return grad1, grad2, loss, mu_out
        
    def transition_and_update( x, y,  part_w1, part_w2, grad1, grad2, first = False):
        shape_w1= grad1.shape 
        shape_w2= grad2.shape    
        if first:
            W1 = cnn_model.layers[0].get_weights()[0]
            W2 = cnn_model.layers[3].get_weights()[0]  
            W1 = tf.convert_to_tensor(W1)
            W2 = tf.convert_to_tensor(W2) 
            part_w1 = tf.random.normal([kernels_size[0],kernels_size[0], 1, num_kernels[0], N], stddev = init_sigma) # creating particles layer 1
            part_w2 = tf.random.normal([N, shape_w2[0],shape_w2[1]], stddev = init_sigma) # creating particles layer 2
            W1_exp = tf.expand_dims(W1, axis =-1)
            W1_reshape = tf.broadcast_to(W1_exp, part_w1.shape)
            W2_exp = tf.expand_dims(W2, axis = 0)
            W2_reshape = tf.broadcast_to(W2_exp, part_w2.shape)
            part_w1 = tf.math.add(part_w1 ,W1_reshape)
            part_w2 = tf.math.add(part_w2 ,W2_reshape)

        grad1 = tf.expand_dims(grad1, axis =-1) 
        grad1_exp = tf.broadcast_to(grad1, part_w1.shape)
        part_w1 = part_w1 - lr* grad1_exp
        grad2 = tf.expand_dims(grad2, axis =0) 
        grad2_exp = tf.broadcast_to(grad2, part_w2.shape)
        part_w2 = part_w2 - lr* grad2_exp

        w2_estimate = tf.reduce_mean(part_w2, axis=0)
        part_w1 = BaSIS_Conv_likelihood_MNIST( x, part_w1, w2_estimate,  y, N = N, weights = weights,sigma_part =sigma_particles_conv)
        w1_estimate = tf.reduce_mean(part_w1, axis=-1)
        w1_estimate_resh = tf.reshape( w1_estimate, shape_w1)
        w1_numpy = w1_estimate_resh.numpy()
        cnn_model.layers[0].set_weights([w1_numpy])
        
        part_w2, y_out = BaSIS_Flatten_and_FC_MNIST( x, w1_estimate, part_w2,  y, N=N, weights = weights,sigma_part =sigma_particles_fc) # y_out shape =[batch,N,units]
        w2_estimate = tf.reduce_mean(part_w2, axis=0)
        w2_estimate = tf.reshape( w2_estimate, shape_w2)
        w2_numpy = w2_estimate.numpy()
        cnn_model.layers[3].set_weights([w2_numpy])      
        return  y_out, part_w1, part_w2
    @tf.function
    def validation_on_batch(x, y, w1, w2):                     
        mu_out, fc = cnn_model(x, training=False) 
        cnn_model.trainable = False              
        vloss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = fc, labels = y),axis=0)                                          
        total_vloss = vloss    
        particles_out, y_out, fc = CNN_model_test_MNIST(x, w1, w2, N= N)
        return total_vloss, mu_out, particles_out, y_out
    @tf.function
    def test_on_batch(x, y, w1, w2):  
        cnn_model.trainable = False                    
        mu_out, fc = cnn_model(x, training=False)  
        particles_out, y_out, fc = CNN_model_test_MNIST(x, w1, w2, N=N)          
        return mu_out,particles_out, y_out, fc
    @tf.function    
    def create_adversarial_pattern(input_image, input_label):
          with tf.GradientTape() as tape:
            tape.watch(input_image)
            cnn_model.trainable = False 
            prediction, fc = cnn_model(input_image) 
            loss= tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = fc, labels = y),axis=0)                           
            
          # Get the gradients of the loss w.r.t to the input image.
          gradient = tape.gradient(loss, input_image)
          # Get the sign of the gradients to create the perturbation
          signed_grad = tf.sign(gradient)
          return signed_grad 
               
    if Training:
        first_step = True
        part_w1 = 0 
        part_w2 = 0
        if continue_training:
            first_step = False
            saved_model_path = '/saved_models/epoch_{}/{}_particles/'.format(saved_model_epochs, N)
            cnn_model.load_weights(saved_model_path + 'vdp_cnn_model')
            init_1 = open(PATH +'particles.pkl'.format(epochs,N), 'rb')
            part_w1, part_w2 = pickle.load(init_1)
            init_1.close()
        
        train_acc = np.zeros(epochs)
        train_acc2 = np.zeros(epochs)  
        valid_acc = np.zeros(epochs)
        valid_acc2 = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        
        start = timeit.default_timer()       
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)           
            acc1 = 0 
            acc2 = 0 
            acc_valid1 = 0
            acc_valid2 = 0 
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0           
            #-------------Training--------------------
            for step, (x, y) in enumerate(tr_dataset):                         
                update_progress(step/int(x_train.shape[0]/(batch_size)) )                
                grad1,grad2, loss, mu_out = get_gradient(x, y) 
                y_out, part_w1, part_w2 = transition_and_update(x, y,part_w1,part_w2, grad1, grad2, first = first_step)           #y_out shape batch,N,units            
                first_step = False
                err1+= loss.numpy() 
                y_out = tf.reduce_mean(y_out, axis = 1)
                
                correct = tf.equal(tf.cast(tf.argmax(mu_out,-1),tf.int32),tf.cast(y,tf.int32))
                accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
            
                mu_out2, particles_out, y_out, fc   = test_on_batch(x, y, part_w1, part_w2)
                accuracy2, particles_prediction = accuracy_test(particles_out, y)
                acc2+=accuracy2.numpy()
                acc1+=accuracy.numpy()        
                        
                if step % 50 == 0:
                    
                    print("\n Step:", step, "Loss:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy from model so far: %.3f" % float(acc1/(tr_no_steps + 1.)))     
                    print("Total Training accuracy from particles so far: %.3f" % float(acc2/(tr_no_steps + 1.)))                                                                
                tr_no_steps+=1 
       
            train_acc[epoch] = acc1/tr_no_steps
            train_acc2[epoch] = acc2/tr_no_steps
            train_err[epoch] = err1/tr_no_steps        
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])         
            #---------------Validation----------------------                  
            for step, (x, y) in enumerate(val_dataset):               
                update_progress(step / int(x_test.shape[0] / (batch_size)) )   
                total_vloss, mu_out, particles_out, y_out  = validation_on_batch(x, y, part_w1, part_w2 )                
                
                err_valid1+= total_vloss.numpy()                               
                correct = tf.equal(tf.cast(tf.argmax(mu_out,-1),tf.int32),tf.cast(y,tf.int32))
                va_accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
            
                y=tf.cast(y,tf.int64)
                acc_valid1+=va_accuracy.numpy() 
                
                corr2 = tf.equal(tf.math.argmax(y_out, axis=-1),y)
                va_accuracy2 = tf.reduce_mean(tf.cast(corr2,tf.float32))
                acc_valid2+=va_accuracy2.numpy() 
                if step % 50 == 0:                   
                    print("Step:", step, "Loss:", float(total_vloss))
                    print("Total validation accuracy from model so far: %.3f" % va_accuracy)  
                    print("Total validation accuracy from particles so far: %.3f" % va_accuracy2)             
                va_no_steps+=1
   
            valid_acc[epoch] = acc_valid1/va_no_steps  
            valid_acc2[epoch] = acc_valid2/va_no_steps      
            valid_error[epoch] = err_valid1/va_no_steps
            stop = timeit.default_timer()
            cnn_model.save_weights(PATH + 'vdp_cnn_model')   

            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch], train_acc2[epoch])
            print('Validation Acc  ', valid_acc[epoch], valid_acc2[epoch])           
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])           
        #-----------------End Training--------------------------             
        cnn_model.save_weights(PATH + 'vdp_cnn_model')        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation CNN on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_CNN_on_MNIST_Data_acc.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')            
            plt.title("Density Propagation CNN on MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_CNN_on_MNIST_Data_error.png')
            plt.close(fig)
        
        f = open(PATH + 'training_validation_acc_error.pkl', 'wb')         
        pickle.dump([train_acc, valid_acc, train_err, valid_error], f)                                                   
        f.close()                  

        f2 = open(PATH + 'particles.pkl', 'wb')         
        pickle.dump([part_w1.numpy(), part_w2.numpy()], f2)                                                   
        f2.close()

        textfile = open(PATH + 'Related_hyperparameters.txt','w')    
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n regularization factor : ' +str(reg_factor)) 
        textfile.write('\n sigma for particles convolution : ' +str(sigma_particles_conv))
        textfile.write('\n sigma for particles fc : ' +str(sigma_particles_fc))
        textfile.write('\n sigma to initialize particles  : ' +str(init_sigma))
        
        if weights == 'inv' :       
            textfile.write('\n importance ratios weights computed as 1/CE ') 
        else:
            textfile.write('\n importance ratios weights computed as exp(-CE) ') 
        textfile.write("\n---------------------------------")          
        if Training: 
            textfile.write('\n Total run time in sec : ' +str(stop - start))
            if(epochs == 1):
                textfile.write("\n Averaged Training  Accuracy : "+ str( train_acc))
                textfile.write("\n Averaged Validation Accuracy : "+ str(valid_acc ))
                    
                textfile.write("\n Averaged Training  error : "+ str( train_err))
                textfile.write("\n Averaged Validation error : "+ str(valid_error ))
            else:
                textfile.write("\n Averaged Training  Accuracy : "+ str(np.mean(train_acc[epoch])))
                textfile.write("\n Averaged Validation Accuracy : "+ str(np.mean(valid_acc[epoch])))

                textfile.write("\n Averaged Training  Accuracy from particles : "+ str(np.mean(train_acc2[epoch])))
                textfile.write("\n Averaged Validation Accuracy from particles: "+ str(np.mean(valid_acc2[epoch])))
                
                textfile.write("\n Averaged Training  error : "+ str(np.mean(train_err[epoch])))
                textfile.write("\n Averaged Validation error : "+ str(np.mean(valid_error[epoch])))
        textfile.write("\n---------------------------------")                
        textfile.write("\n---------------------------------")    
        textfile.close()
    #-------------------------Testing-----------------------------    
    elif(Testing):
        test_path = 'test_results/'
        cnn_model.load_weights(PATH + 'vdp_cnn_model') 
        test_no_steps = 0        
        init_1 = open(PATH +'particles.pkl'.format(epochs,N), 'rb')
        part_w1, part_w2 = pickle.load(init_1)
        init_1.close()
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])

        var_out = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        var = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        var_out_fc = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        var_fc = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        acc_test2 = np.zeros(int(x_test.shape[0] / (batch_size)))
        third_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        fourth_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        all_particles = np.zeros([int(x_test.shape[0] / (batch_size)),N, batch_size, class_num])
        variance_correct = []
        variance_incorrect = []
        skew_correct = []
        skew_incorrect = []
        kurt_correct = []
        kurt_incorrect = []
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
            true_x[test_no_steps, :, :, :,:] = x
            true_y[test_no_steps, :] = y
            start = timeit.default_timer()
            mu_out, particles_out, y_out, fc   = test_on_batch(x, y, part_w1, part_w2)              
            stop = timeit.default_timer()
            mu_out_[test_no_steps,:,:] = y_out
            pf = particles_out.numpy()  
            all_particles [test_no_steps,:,:,: ] = pf
            pf_fc = fc.numpy()   
            v = np.var(pf, axis=0)  
            v_fc  = np.var(pf_fc, axis = 0)  
            skew = scipy.stats.skew(pf, axis=0)
            kurt = scipy.stats.kurtosis(pf, axis=0, fisher=False)
            
            var_out[test_no_steps,:,:] = v
            var_out_fc[test_no_steps,:,:] = v_fc
            accuracy = accuracy_test2(particles_out, y)
            accuracy2, particles_prediction = accuracy_test(particles_out, y)
            
        
            y =y.numpy()
            for j in range(batch_size):
                predicted_out = np.argmax(mu_out_[test_no_steps,j,:])
                var[test_no_steps,j] = var_out[test_no_steps,j, int(predicted_out)] 
                var_fc[test_no_steps,j] = var_out_fc[test_no_steps,j, int(predicted_out)] 
                third_moment[test_no_steps,j] = skew[j, int(predicted_out)]
                fourth_moment[test_no_steps,j] = kurt[j, int(predicted_out)]
                if particles_prediction[j]==y[j]:
                   
                    variance_correct.append(var_out[test_no_steps,j, int(predicted_out)] )
                    skew_correct.append(skew[j, int(predicted_out)] )
                    kurt_correct.append(kurt[j, int(predicted_out)] )
                else:
                    
                    variance_incorrect.append(var_out[test_no_steps,j, int(predicted_out)] )
                    skew_incorrect.append(skew[j, int(predicted_out)] )
                    kurt_incorrect.append(kurt[j, int(predicted_out)] )

                    
            acc_test[test_no_steps] = accuracy.numpy()
            acc_test2[test_no_steps] = accuracy2.numpy()
             
            if step % 100 == 0:
                print("Total running accuracy so far: %.3f" % accuracy.numpy())
                print("Total running accuracy2 so far: %.3f" % accuracy2.numpy())             
            test_no_steps+=1      
             
        test_acc = np.mean(acc_test) 
        test_acc2 = np.mean(acc_test2)          
        print('Test accuracy : ', test_acc,test_acc2 )                  
        
        if not os.path.exists(PATH+ test_path ):
            os.makedirs(PATH+ test_path )
        p = open(PATH + test_path + 'particles_output.pkl', 'wb')      
        pickle.dump([pf, y ], p)                                                  
        p.close()
        
        print('Output Variance', np.mean(var))
        z_test,p_value = hypothesis_test(variance_correct,variance_incorrect)
        z_test1,p_value1 = hypothesis_test1(variance_correct,variance_incorrect)
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(reg_factor))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : "+ str( test_acc)) 
        textfile.write("\n Test Accuracy : "+ str( test_acc2))  
        textfile.write('\n Total run time in sec : ' +str(stop - start)) 
                            
        textfile.write("\n---------------------------------")
        textfile.write("\n Output Variance: "+ str(np.mean(var)))   
        textfile.write("\n Output Variance for correct classified: "+ str(np.mean(variance_correct)))  
        textfile.write("\n Output Variance for incorrect: "+ str(np.mean(variance_incorrect)))   
        textfile.write("\n---------------------------------")
        textfile.write("\n Skew moment: "+ str(np.mean(third_moment)))  
        textfile.write("\n Kurtosis moment"+ str(np.mean(fourth_moment))) 
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Skew for correct: "+ str(np.mean(skew_correct)))
        textfile.write("\n Kurtosis for correct: "+ str(np.mean(kurt_correct)))  
        textfile.write("\n Skew for incorrect: "+ str(np.mean(skew_incorrect)))  
        textfile.write("\n Kurtosis for incorrect: "+ str(np.mean(kurt_incorrect)))  
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Abs of Skew moment: "+ str(np.mean(np.absolute(third_moment))))  
        textfile.write("\n Abs of Kurtosis moment"+ str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Abs of Skew for correct: "+ str(np.mean(np.absolute(skew_correct))))
        textfile.write("\n Abs of Kurtosis for correct: "+ str(np.mean(np.absolute(kurt_correct)))  )
        textfile.write("\n Abs of Skew for incorrect: "+ str(np.mean(np.absolute(skew_incorrect))) ) 
        textfile.write("\n Abs of Kurtosis for incorrect: "+ str(np.mean(np.absolute(kurt_incorrect)))  )
        textfile.write("\n---------------------------------")
        textfile.write("\n Test of hypothesis - z: "+ str( z_test1)) 
        textfile.write("\n Test of hypothesis: "+ str( p_value1)) 
        textfile.write("\n---------------------------------")
        textfile.write("\n Test of hypothesis - t: "+ str( z_test)) 
        textfile.write("\n Test of hypothesis: "+ str( p_value)) 
        textfile.close()

        textfile = open(PATH + test_path + 'Results_for_excel.txt','w')   
              
        textfile.write("\n-------------Accuracy--------------------")
        textfile.write("\n"+ str( test_acc)) 
        textfile.write( "\n"+str( test_acc2))  
                         
        textfile.write("\n-------------Variance /correct/incorrect--------------------")
        textfile.write("\n"+ str(np.mean(var)))   
        textfile.write("\n"+str(np.mean(variance_correct)))  
        textfile.write("\n"+ str(np.mean(variance_incorrect)))   
        textfile.write("\n----------Skew/correct/incorrect-----------------------")
        textfile.write("\n"+ str(np.mean(third_moment)))  
        textfile.write("\n"+str(np.mean(skew_correct)))
        textfile.write("\n"+ str(np.mean(skew_incorrect))) 
        
        textfile.write("\n----------Kurtosis-----------------------")
        textfile.write("\n"+ str(np.mean(fourth_moment))) 
        textfile.write("\n"+ str(np.mean(kurt_correct)))  
        textfile.write("\n"+ str(np.mean(kurt_incorrect)))  
        textfile.write("\n-------------abs skew--------------------")
        textfile.write("\n"+ str(np.mean(np.absolute(third_moment))))  
        textfile.write("\n"+ str(np.mean(np.absolute(skew_correct))))
        textfile.write("\n"+str(np.mean(np.absolute(skew_incorrect))) ) 
        
        textfile.write("\n--------------abs kurtosis-------------------")
       
        textfile.write("\n"+ str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write("\n"+ str(np.mean(np.absolute(kurt_correct)))  )
        
        textfile.write( str(np.mean(np.absolute(kurt_incorrect)))  )
        textfile.write("\n---------------------------------")
        textfile.close()
        if histogram:
                all_particles = np.reshape(np.transpose(all_particles, [0,2,1,3]), [-1, N, class_num])
                true_x = np.reshape(np.squeeze(true_x), [-1, input_dim, input_dim])
                true_y = np.reshape(true_y, -1)
                var = np.reshape(var, -1)
                pred = np.reshape(np.argmax(mu_out_,-1),-1)
                third_moment = np.reshape(third_moment, -1)
                fourth_moment = np.reshape(fourth_moment, -1)
                histogram_with_label(PATH + test_path, 10, 13, true_x,true_y,pred, var, third_moment,fourth_moment, all_particles)
        
    elif(Random_noise):
        test_path = 'test_results_random_noise_{}/'.format(gaussian_noise_var)
        cnn_model.load_weights(PATH + 'vdp_cnn_model') 
        test_no_steps = 0        
        init_1 = open(PATH +'particles.pkl'.format(epochs,N), 'rb')
        part_w1, part_w2 = pickle.load(init_1)
        init_1.close()
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])

        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        var_out = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        covar_out = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num, class_num])
        var = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        var_out_fc = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        var_fc = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        acc_test2 = np.zeros(int(x_test.shape[0] / (batch_size)))
        third_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        fourth_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        all_particles = np.zeros([int(x_test.shape[0] / (batch_size)),N, batch_size, class_num])
        
        variance_correct = []
        variance_incorrect = []
        skew_correct = []
        skew_incorrect = []
        kurt_correct = []
        kurt_incorrect = []
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
            true_y[test_no_steps, :] = y
            noise_free = x
            if Random_noise:
                gaussain_noise_std = gaussian_noise_var**(0.5)
                noise = tf.random.normal(shape = [batch_size, input_dim, input_dim, 1], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype) 
                x = x +  noise
                x = tf.clip_by_value(x, 0.0,1.0)
            true_x[test_no_steps, :, :, :,:] = x
            mu_out, particles_out, y_out,fc   = test_on_batch(x, y, part_w1, part_w2)              
            mu_out_[test_no_steps,:,:] = y_out
            pf = particles_out.numpy()   
            all_particles [test_no_steps,:,:,: ] = pf
            v = np.var(pf, axis=0)
            skew = scipy.stats.skew(pf, axis=0)
            kurt = scipy.stats.kurtosis(pf, axis=0, fisher=False)
            pf_fc = fc.numpy()   
            v_fc = np.var(pf_fc, axis=0)   
            var_out_fc[test_no_steps,:,:] = v_fc
            var_out[test_no_steps,:,:] = v    
            accuracy = accuracy_test2(particles_out, y)

            accuracy2, particles_prediction = accuracy_test(particles_out, y)
            acc_test[test_no_steps] = accuracy.numpy()
            acc_test2[test_no_steps] = accuracy2.numpy()
            y =y.numpy()

            
            for j in range(batch_size):
                vv =  np.cov(pf[:, j,:], rowvar=False, ddof=0)
                covar_out[test_no_steps,j,:,:] = vv  
                snr_signal[step,j] = 10*np.log10( np.sum(np.square(noise_free[j,:,:, :]))/np.sum( np.square(noise_free[j,:,:, :]-x[j,:,:,:]) ))
                predicted_out = np.argmax(mu_out_[test_no_steps,j,:])
                var[test_no_steps,j] = var_out[test_no_steps,j, int(predicted_out)] 
                var_fc[test_no_steps,j] = var_out_fc[test_no_steps,j, int(predicted_out)] 
                third_moment[test_no_steps,j] = skew[j, int(predicted_out)]
                fourth_moment[test_no_steps,j] = kurt[j, int(predicted_out)]
                if particles_prediction[j]==y[j]:
                    
                    variance_correct.append(var_out[test_no_steps,j, int(predicted_out)] )
                    skew_correct.append(skew[j, int(predicted_out)] )
                    kurt_correct.append(kurt[j, int(predicted_out)] )
                else:
                    
                    variance_incorrect.append(var_out[test_no_steps,j, int(predicted_out)] )
                    skew_incorrect.append(skew[j, int(predicted_out)] )
                    kurt_incorrect.append(kurt[j, int(predicted_out)] )
            
            if step % 100 == 0:
                print("Total running accuracy so far: %.3f" % accuracy.numpy())
                print("Total running accuracy2 so far: %.3f" % accuracy2.numpy())             
            test_no_steps+=1      
             
        test_acc = np.mean(acc_test) 
        test_acc2 = np.mean(acc_test2)          
        print('Test accuracy : ', test_acc,test_acc2 )                  
        if not os.path.exists(PATH+ test_path ):
            os.makedirs(PATH+ test_path )
        p = open(PATH + test_path + 'particles_output.pkl', 'wb')      
        pickle.dump([pf ], p)                                                  
        p.close()

        p2 = open(PATH + test_path + 'input_output.pkl', 'wb')      
        pickle.dump([x,y], p2)                                                  
        p2.close()
        v = np.mean(var)
        s = np.mean(snr_signal)
        pf = open(PATH + test_path + 'acc_var_snr.pkl', 'wb')                   
        pickle.dump([test_acc,  v, s], pf)                                                  
        pf.close() 
        
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))    
        z_test,p_value = hypothesis_test(variance_correct,variance_incorrect)
        z_test1,p_value1 = hypothesis_test1(variance_correct,variance_incorrect)
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))        
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end)) 
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(reg_factor))        
        textfile.write("\n---------------------------------")
        textfile.write("\n Test Accuracy : "+ str( test_acc)) 
        textfile.write("\n Test Accuracy : "+ str( test_acc2))   
                         
        textfile.write("\n---------------------------------")
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussian_noise_var ))   
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))           
        textfile.write("\n---------------------------------")    
        textfile.write("\n Output Variance: "+ str(np.mean(var))) 
        textfile.write("\n Output Variance for correct classified: "+ str(np.mean(variance_correct)))  
        textfile.write("\n Output Variance for incorrect: "+ str(np.mean(variance_incorrect)))   
        
        textfile.write("\n---------------------------------")
        textfile.write("\n Skew moment: "+ str(np.mean(third_moment)))  
        textfile.write("\n : Kurtosis "+ str(np.mean(fourth_moment)))
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Kurtosis for correct: "+ str(np.mean(kurt_correct)))  
        textfile.write("\n Skew for correct: "+ str(np.mean(skew_correct))) 
        textfile.write("\n Kurtosis for incorrect: "+ str(np.mean(kurt_incorrect)))  
        textfile.write("\n Skew for incorrect: "+ str(np.mean(skew_incorrect))) 
        textfile.write("\n---------------------------------")
        textfile.write("\n Abs of Skew moment: "+ str(np.mean(np.absolute(third_moment))))  
        textfile.write("\n Abs of Kurtosis moment"+ str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Abs of Skew for correct: "+ str(np.mean(np.absolute(skew_correct))))
        textfile.write("\n Abs of Kurtosis for correct: "+ str(np.mean(np.absolute(kurt_correct)))  )
        textfile.write("\n Abs of Skew for incorrect: "+ str(np.mean(np.absolute(skew_incorrect))) ) 
        textfile.write("\n Abs of Kurtosis for incorrect: "+ str(np.mean(np.absolute(kurt_incorrect)))  )
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Test of hypothesis - z: "+ str( z_test1)) 
        textfile.write("\n Test of hypothesis: "+ str( p_value1)) 
        textfile.write("\n---------------------------------")
        textfile.write("\n Test of hypothesis - t: "+ str( z_test)) 
        textfile.write("\n Test of hypothesis: "+ str( p_value)) 
        textfile.close()
        textfile = open(PATH + test_path + 'Results_for_excel.txt','w')   
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussian_noise_var ))   
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))      
        textfile.write("\n-------------Accuracy--------------------")
        textfile.write( "\n"+str( test_acc)) 
        textfile.write( "\n"+str( test_acc2))  
                         
        textfile.write("\n-------------Variance /correct/incorrect--------------------")
        textfile.write("\n"+ str(np.mean(var)))   
        textfile.write("\n"+str(np.mean(variance_correct)))  
        textfile.write("\n"+ str(np.mean(variance_incorrect)))   
        textfile.write("\n----------Skew/correct/incorrect-----------------------")
        textfile.write( "\n"+str(np.mean(third_moment)))  
        textfile.write("\n"+str(np.mean(skew_correct)))
        textfile.write("\n"+ str(np.mean(skew_incorrect))) 
        
        textfile.write("\n----------Kurtosis-----------------------")
        textfile.write("\n"+ str(np.mean(fourth_moment))) 
        textfile.write("\n"+ str(np.mean(kurt_correct)))  
        textfile.write("\n"+ str(np.mean(kurt_incorrect)))  
        textfile.write("\n-------------abs skew--------------------")
        textfile.write( "\n"+str(np.mean(np.absolute(third_moment))))  
        textfile.write("\n"+ str(np.mean(np.absolute(skew_correct))))
        textfile.write("\n"+str(np.mean(np.absolute(skew_incorrect))) ) 
        
        textfile.write("\n--------------abs kurtosis-------------------")
       
        textfile.write( "\n"+str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write("\n"+ str(np.mean(np.absolute(kurt_correct)))  )
        
        textfile.write("\n"+ str(np.mean(np.absolute(kurt_incorrect)))  )
        textfile.write("\n---------------------------------")
        textfile.close()
        if histogram:
                all_particles = np.reshape(np.transpose(all_particles, [0,2,1,3]), [-1, N, class_num])
                true_x = np.reshape(np.squeeze(true_x), [-1, input_dim, input_dim])
                true_y = np.reshape(true_y,-1)
                var = np.reshape(var, -1)
                pred = np.reshape(np.argmax(mu_out_,-1),-1)
                third_moment = np.reshape(third_moment, -1)
                fourth_moment = np.reshape(fourth_moment, -1)
                histogram_with_label(PATH + test_path, 10, 13, true_x,true_y,pred, var, third_moment,fourth_moment, all_particles)
        colormap = True
        if colormap:
            covar_out = np.reshape(covar_out, [-1,class_num, class_num ])
            create_heatmap(PATH + test_path,covar_out)
    
    elif(Adversarial_noise):
        init_1 = open(PATH +'particles.pkl'.format(epochs,N), 'rb')
        part_w1, part_w2 = pickle.load(init_1)
        init_1.close()
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(epsilon)            
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)              
        cnn_model.load_weights(PATH + 'vdp_cnn_model')       
        test_no_steps = 0        
        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 1])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        var = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        third_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        fourth_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        all_particles = np.zeros([int(x_test.shape[0] / (batch_size)),N, batch_size, class_num])
        sequence = [i for i in range(int(x_test.shape[0]))]
        numbers = []
        
        variance_correct = []
        variance_incorrect = []
        skew_correct = []
        skew_incorrect = []
        kurt_correct = []
        kurt_incorrect = []
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
            #true_x[test_no_steps, :, :, :,:] = x
            true_y[test_no_steps, :] = y
            
            if Targeted:
                y_hot = tf.one_hot(y, depth=class_num)
                y_true_batch = np.zeros_like(y_hot)
                y_true_batch[:, adversary_target_cls] = 1.0            
                adv_perturbations[test_no_steps, :, :, :,:] = create_adversarial_pattern(x, y_true_batch)
            else:
                adv_perturbations[test_no_steps, :, :, :,:] = create_adversarial_pattern(x, y)
            adv_x = x + epsilon*adv_perturbations[test_no_steps, :, :, :,:] 
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0) 
            true_x[test_no_steps, :, :, :,:] = adv_x
            mu_out, particles_out, y_out,fc   = test_on_batch(adv_x, y,  part_w1, part_w2)           
            mu_out_[test_no_steps,:,:] = y_out
            pf = particles_out.numpy() 
            all_particles [test_no_steps,:,:,: ] = pf  
            v = np.var(pf, axis=0)
            skew = scipy.stats.skew(pf, axis=0)
            kurt = scipy.stats.kurtosis(pf, axis=0, fisher=False)
            sigma_[test_no_steps, :, :]= v           
            accuracy, particles_prediction = accuracy_test(particles_out, y)
            
            
            acc_test[test_no_steps]=accuracy.numpy()
            if step % 10 == 0:

                print("Total running accuracy so far: %.3f" % accuracy.numpy())                           
            y =y.numpy()
            for j in range(batch_size):               
                predicted_out = np.argmax(mu_out_[test_no_steps,j,:])
                var[test_no_steps,j] = sigma_[test_no_steps,j, int(predicted_out)]
                snr_signal[step,j] = 10*np.log10( np.sum(np.square(x[j,:,:, :]))/np.sum( np.square(x[j,:,:, :] - adv_x[j,:,:, :]  ) ))
                third_moment[test_no_steps,j] = skew[j, int(predicted_out)]
                fourth_moment[test_no_steps,j] = kurt[j, int(predicted_out)]
                if particles_prediction[j]==y[j]:
                    
                    variance_correct.append(sigma_[test_no_steps,j, int(predicted_out)] )
                    skew_correct.append(skew[j, int(predicted_out)] )
                    kurt_correct.append(kurt[j, int(predicted_out)] )
                else:
                   
                    variance_incorrect.append(sigma_[test_no_steps,j, int(predicted_out)] )
                    skew_incorrect.append(skew[j, int(predicted_out)] )
                    kurt_incorrect.append(kurt[j, int(predicted_out)] )
                    wrong_ind = sequence[step]
                    numbers.append(wrong_ind)

            test_no_steps+=1 
        test_acc = np.mean(acc_test)         
        print('Test accuracy : ', test_acc)                       
        if not os.path.exists(PATH+ test_path ):
            os.makedirs(PATH+ test_path )
        
        pf2 = open(PATH + test_path + 'uncertainty_info.pkl', 'wb')            
        pickle.dump([pf], pf2)                                                
        pf2.close()
        
        p2 = open(PATH + test_path + 'input_output.pkl', 'wb')      
        pickle.dump([adv_x,y ], p2)                                                  
        p2.close()
        v = np.mean(var)
        s = np.mean(snr_signal)
        pf = open(PATH + test_path + 'acc_var_snr.pkl', 'wb')                   
        pickle.dump([test_acc,  v, s], pf)                                                  
        pf.close() 
        
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))         
        
        
        textfile = open(PATH + test_path + 'Related_hyperparameters.txt','w')   
        textfile.write(' Input Dimension : ' +str(input_dim))
        textfile.write('\n No of Kernels : ' +str(num_kernels))
        textfile.write('\n Number of Classes : ' +str(class_num))
        textfile.write('\n No of epochs : ' +str(epochs))
        textfile.write('\n Initial Learning rate : ' +str(lr)) 
        textfile.write('\n Ending Learning rate : ' +str(lr_end))  
        textfile.write('\n kernels Size : ' +str(kernels_size))  
        textfile.write('\n Max pooling Size : ' +str(maxpooling_size)) 
        textfile.write('\n Max pooling stride : ' +str(maxpooling_stride))
        textfile.write('\n batch size : ' +str(batch_size)) 
        textfile.write('\n KL term factor : ' +str(reg_factor))      
        textfile.write("\n---------------------------------")
        textfile.write("\n Averaged Test Accuracy : "+ str( test_acc))  
        textfile.write("\n Output Variance: "+ str(np.mean(var))) 
        textfile.write("\n Output Variance for correct classified: "+ str(np.mean(variance_correct)))  
        textfile.write("\n Output Variance for incorrect: "+ str(np.mean(variance_incorrect)))                    
        textfile.write("\n---------------------------------")
        
        textfile.write("\n Skew moment: "+ str(np.mean(third_moment)))  
        textfile.write("\n : Kurtosis "+ str(np.mean(fourth_moment)))  
        textfile.write("\n---------------------------------")
        textfile.write("\n Kurtosis for correct: "+ str(np.mean(kurt_correct)))  
        textfile.write("\n Skew for correct: "+ str(np.mean(skew_correct))) 
        textfile.write("\n Kurtosis for incorrect: "+ str(np.mean(kurt_incorrect)))  
        textfile.write("\n Skew for incorrect: "+ str(np.mean(skew_incorrect))) 
        textfile.write("\n---------------------------------")
        textfile.write("\n Abs of Skew moment: "+ str(np.mean(np.absolute(third_moment))))  
        textfile.write("\n Abs of Kurtosis moment"+ str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write("\n---------------------------------")
        textfile.write("\n---------------------------------")
        textfile.write("\n Abs of Skew for correct: "+ str(np.mean(np.absolute(skew_correct))))
        textfile.write("\n Abs of Kurtosis for correct: "+ str(np.mean(np.absolute(kurt_correct)))  )
        textfile.write("\n Abs of Skew for incorrect: "+ str(np.mean(np.absolute(skew_incorrect))) ) 
        textfile.write("\n Abs of Kurtosis for incorrect: "+ str(np.mean(np.absolute(kurt_incorrect)))  )
        textfile.write("\n---------------------------------")

        if Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))                   
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))    
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
        textfile.write("\n---------------------------------")    
        textfile.close() 
        textfile = open(PATH + test_path + 'Results_for_excel.txt','w')   
        if Adversarial_noise:
              
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))      
        textfile.write("\n-------------Accuracy--------------------")
        textfile.write( "\n"+str( test_acc))                 
        textfile.write("\n-------------Variance /correct/incorrect--------------------")
        textfile.write("\n"+ str(np.mean(var)))   
        textfile.write("\n"+str(np.mean(variance_correct)))  
        textfile.write("\n"+ str(np.mean(variance_incorrect)))   
        textfile.write("\n----------Skew/correct/incorrect-----------------------")
        textfile.write( "\n"+str(np.mean(third_moment)))  
        textfile.write("\n"+str(np.mean(skew_correct)))
        textfile.write("\n"+ str(np.mean(skew_incorrect))) 
        
        textfile.write("\n----------Kurtosis-----------------------")
        textfile.write( "\n"+str(np.mean(fourth_moment))) 
        textfile.write( "\n"+str(np.mean(kurt_correct)))  
        textfile.write( "\n"+str(np.mean(kurt_incorrect)))  
        textfile.write("\n-------------abs skew--------------------")
        textfile.write( "\n"+str(np.mean(np.absolute(third_moment))))  
        textfile.write("\n"+ str(np.mean(np.absolute(skew_correct))))
        textfile.write("\n"+str(np.mean(np.absolute(skew_incorrect))) ) 
        
        textfile.write("\n--------------abs kurtosis-------------------")
       
        textfile.write( "\n"+str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write( "\n"+str(np.mean(np.absolute(kurt_correct)))  )
        
        textfile.write( "\n"+str(np.mean(np.absolute(kurt_incorrect)))  )
        textfile.write("\n---------------------------------")
        textfile.close()
        if histogram:
                all_particles = np.reshape(np.transpose(all_particles, [0,2,1,3]), [-1, N, class_num])
                true_x = np.reshape(np.squeeze(true_x), [-1, input_dim, input_dim])
                true_y = np.reshape(true_y,-1)
                var = np.reshape(var, -1)
                pred = np.reshape(np.argmax(mu_out_,-1),-1)
                third_moment = np.reshape(third_moment, -1)
                fourth_moment = np.reshape(fourth_moment, -1)
                histogram_with_label(PATH + test_path, 10, 13, true_x,true_y,pred, var, third_moment,fourth_moment, all_particles) 

                histogram_adv(PATH + test_path, 10, numbers,true_x,true_y,pred, var, third_moment,fourth_moment, all_particles)
    
    
if __name__ == '__main__': 
    main_function(folder, input_dim, num_kernels, kernels_size, maxpooling_size, maxpooling_stride, maxpooling_pad, class_num, batch_size,
        epochs, lr, lr_end, reg_factor, N,sigma_particles_conv,sigma_particles_fc, init_sigma, 
        Training , continue_training ,  saved_model_epochs, Testing , weights,
        Random_noise, gaussian_noise_var, Adversarial_noise, epsilon, adversary_target_cls, Targeted,
         histogram)
    
    