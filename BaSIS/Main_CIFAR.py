
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import timeit
import scipy
from scipy import stats
from scipy.stats import skewtest
from scipy.stats import kurtosistest
from mpl_toolkits import axes_grid1
from scipy.io import loadmat
plt.ioff()
cifar10 = tf.keras.datasets.cifar10

from model_functions import *
from other_functions import *
import argparse


# Initialize argparse
parser = argparse.ArgumentParser(description='This script is used to train & test the BaSIS-Net model on the CIFAR-10 dataset. Specify the path to save the model, and other hyperparameters.')

# Add arguments
parser.add_argument('--folder', type=str, required=True, 
                    help='Folder to save model and logs')
parser.add_argument('--input_dim', type=int,  default=32, 
                    help='Size of the input - 32 for CIFAR-10')
parser.add_argument('--num_kernels', type=int, nargs='+', default=[32, 32, 32, 32, 64, 64, 64, 128, 128, 128], 
                    help='Number of kernels for convolutional layers')
parser.add_argument('--kernels_size', type=int, nargs='+', default=[5, 3, 3, 3, 3, 3, 3, 3, 3, 1], 
                    help='Kernel size for convolutional layers')
parser.add_argument('--maxpooling_size', type=int, nargs='+', default=[2, 2, 2, 2, 2], 
                    help='Max pooling size')
parser.add_argument('--maxpooling_stride', type=int, nargs='+', default=[2, 2, 2, 2, 2], 
                    help='Stride for max pooling')
parser.add_argument('--maxpooling_pad', type=str, default='SAME', 
                    help='Padding for max pooling')
parser.add_argument('--class_num', type=int, default=10, 
                    help='Number of classes')
parser.add_argument('--batch_size', type=int, default=50, 
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=250, 
                    help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='Learning rate')
parser.add_argument('--lr_end', type=float, default=0.00001, 
                    help='Ending learning rate')
parser.add_argument('--reg_factor', type=float, default=0.003, 
                    help='Regularization factor')
parser.add_argument('--N', type=int, default=100, 
                    help='Number of particles')
parser.add_argument('--sigma_particles', type=float, default=0.00001, 
                    help='Sigma for particles')
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
                    help='Current implementation is for exp - other option inv')
parser.add_argument('--Random_noise', type=bool, default=False, 
                    help='Flag to indicate if random (Gaussian) noise should be added to testing dataset')
parser.add_argument('--gaussian_noise_var', type=float, default=0.01, 
                    help='Variance of the Gaussian noise to be added')
parser.add_argument('--Adversarial_noise', type=bool, default=False, 
                    help='Flag to indicate if (FGSM) adversarial noise should be added to testing dataset')
parser.add_argument('--epsilon', type=float, default=0, 
                    help='Epsilon value for adversarial noise')
parser.add_argument('--PGD_Adversarial_noise', type=bool, default=False, 
                    help='Flag to indicate if PGD adversarial noise should be added to testing dataset')
parser.add_argument('--adversary_target_cls', type=int, default=3, 
                    help='Target class for adversarial noise')
parser.add_argument('--maxAdvStep', type=int, default=40, 
                    help='Maximum number of steps for PGD adversarial noise')
parser.add_argument('--stepSize', type=int, default=3, 
                    help='Step size for PGD adversarial noise')
parser.add_argument('--Targeted', type=bool, default=False, 
                    help='Flag to indicate if the adversarial attack is targeted')
parser.add_argument('--histogram', type=bool, default=False, 
                    help='Flag to indicate if prediction histograms should be generated')
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
sigma_particles = args.sigma_particles
init_sigma = args.init_sigma
Training = args.Training
continue_training = args.continue_training
saved_model_epochs = args.saved_model_epochs
Testing = args.Testing
weights = args.weights
Random_noise = args.Random_noise
gaussian_noise_var = args.gaussian_noise_var
Adversarial_noise = args.Adversarial_noise
PGD_Adversarial_noise = args.PGD_Adversarial



def main_function(input_dim, num_kernels, kernels_size,  maxpooling_size, maxpooling_stride, maxpooling_pad, class_num, batch_size,
        epochs, lr, lr_end , reg_factor, N ,sigma_particles,init_sigma, Testing, weights,
        Random_noise, gaussian_noise_var,Adversarial_noise, epsilon,  adversary_target_cls, Targeted,
        Training, continue_training,  saved_model_epochs, histogram , 
        PGD_Adversarial_noise, maxAdvStep, stepSize):   

    if weights == 'inv'   :
        PATH = folder+'Cifar_10/epoch_{}/{}_particles/inv_ce/'.format( epochs,N) 
    else:   
        PATH = folder+'Cifar_10/epoch_{}/{}_particles/exp_ce/'.format( epochs,N) 
    
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
    x_train, x_test = x_train/255.0, x_test/255.0  
    x_train = x_train.astype('float32')  
    x_test = x_test.astype('float32') 
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    tr_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
    cnn_model = Density_prop_CNN_CIFAR(kernel_size=kernels_size, n_kernels=num_kernels,regularization = reg_factor, pooling_size=maxpooling_size,
                                 pooling_stride=maxpooling_stride, pooling_pad=maxpooling_pad, n_labels=class_num,
                                 name='pf_cnn')
    n_steps = int(x_train.shape[0] /batch_size)
    num_train_steps = epochs * n_steps
 
    @tf.function  # Make it fast.
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
        return gradients, loss,loss_final, mu_out
    def find_lr(step):
        decay = lr/ epochs
        lr_new = lr * 1.0/(1.0 + decay*step)
        return lr_new   
    def find_lr_polynomial(step, power= 2.):
        decay = num_train_steps
        step = min(step, decay)
        lr_new = ((lr-lr_end) * (1.0 - step/decay)**power)+ lr_end
        return lr_new   
    def set_weights(part_w1,shape_w1, i):
        if i <10:
            w1_estimate = tf.reduce_mean(part_w1, axis=-1)
        else:
            
            w1_estimate = tf.reduce_mean(part_w1, axis=0)
        
        w1_numpy = w1_estimate.numpy()
        cnn_model.layers[i].set_weights([w1_numpy])
        return w1_estimate
    
    def first_step_particles(i,  fc = False):
        W = cnn_model.layers[i].get_weights()[0]
        W = tf.convert_to_tensor(W) 
        shape  = W.shape
        if fc:
            part = tf.random.normal([N, shape[0],shape[1]], stddev = init_sigma) # creating particles fc layer 
            W_exp = tf.expand_dims(W, axis = 0)
        else:
            part = tf.random.normal([shape[0],shape[1],shape[2],shape[3], N], stddev = init_sigma) # creating particles 
            W_exp = tf.expand_dims(W, axis =-1)
        W_reshape = tf.broadcast_to(W_exp, part.shape)
        part_w = tf.math.add(part ,W_reshape)
        return part_w
    def backprop(i, part_w, grad, lr,  fc = False):
        gradients = grad[i]
        grad1 = tf.convert_to_tensor(gradients)
        if fc:
            grad1 = tf.expand_dims(grad1, axis = 0 )
        else:
            grad1 = tf.expand_dims(grad1, axis = -1)
        grad1_exp = tf.broadcast_to(grad1, part_w.shape)
        part_w = part_w - lr* grad1_exp
        if fc:
            estimate = tf.reduce_mean(part_w, axis=0)
        else:
            estimate = tf.reduce_mean(part_w, axis=-1)
        return part_w, estimate
    def transition_and_update(x, y, lr, gradient, part_w1, part_w2, part_w3, part_w4, part_w5, part_w6, part_w7, part_w8,  part_w9,  part_w10,  part_w11, first = False):
        if first:
            part_w1 = first_step_particles(0)
            part_w2 = first_step_particles(1)
            part_w3 = first_step_particles(2)
            part_w4 = first_step_particles(3)
            part_w5 = first_step_particles(4)
            part_w6 = first_step_particles(5)
            part_w7 = first_step_particles(6)
            part_w8 = first_step_particles(7)
            part_w9 = first_step_particles(8)
            part_w10 = first_step_particles(9)
            part_w11 = first_step_particles(10, fc = True)
        
        part_w1, W1 = backprop(0, part_w1, gradient, lr)
        part_w2, W2= backprop(1,part_w2, gradient, lr)
        part_w3, W3 = backprop(2,part_w3, gradient, lr)
        part_w4, W4 = backprop(3,part_w4, gradient, lr)
        
        part_w5, W5 = backprop(4,part_w5, gradient, lr)
        part_w6, W6= backprop(5,part_w6, gradient, lr)
        part_w7, W7 = backprop(6,part_w7, gradient, lr )
        part_w8, W8 = backprop(7, part_w8, gradient, lr)
        part_w9, W9 = backprop(8, part_w9, gradient, lr)
        part_w10, W10 = backprop(9, part_w10, gradient, lr)
        part_w11, W11 = backprop(10,part_w11, gradient, lr ,fc = True)

        shape_w1,shape_w2, shape_w3= W1.shape,W2.shape ,W3.shape
        shape_w4, shape_w5, shape_w6,shape_w7 = W4.shape,W5.shape, W6.shape, W7.shape  
        shape_w8,shape_w9, shape_w10, shape_w11= W8.shape,W9.shape ,W10.shape      ,W11.shape                                                                                                       
        part_w1 = BaSIS_Conv_likelihood_CIFAR(1, x, part_w1, W2,W3, W4, W5, W6, W7, W8,W9,W10,W11, y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N,sigma_part = sigma_particles, weights = weights)
        w1_estimate = set_weights(part_w1, shape_w1, 0)
        part_w2  = BaSIS_Conv_likelihood_CIFAR(2, x,w1_estimate, part_w2,W3, W4, W5, W6, W7,W8,W9,W10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N,sigma_part = sigma_particles, weights = weights)
        w2_estimate = set_weights(part_w2, shape_w2, 1)
        part_w3 = BaSIS_Conv_likelihood_CIFAR(3, x,w1_estimate, w2_estimate,part_w3, W4, W5, W6, W7,W8,W9,W10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N,sigma_part = sigma_particles, weights = weights)
        w3_estimate = set_weights(part_w3, shape_w3, 2)
        part_w4 = BaSIS_Conv_likelihood_CIFAR(4, x,w1_estimate, w2_estimate,w3_estimate, part_w4, W5, W6, W7,W8,W9,W10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights)
        w4_estimate = set_weights(part_w4, shape_w4, 3)
        part_w5  = BaSIS_Conv_likelihood_CIFAR(5, x,w1_estimate, w2_estimate,w3_estimate, w4_estimate, part_w5, W6, W7,W8,W9,W10,W11, y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights)
        w5_estimate = set_weights(part_w5, shape_w5, 4)
        part_w6  = BaSIS_Conv_likelihood_CIFAR(6, x,w1_estimate, w2_estimate,w3_estimate, w4_estimate, w5_estimate, part_w6, W7,W8,W9,W10,W11, y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights)
        w6_estimate = set_weights(part_w6, shape_w6, 5)
        part_w7=   BaSIS_Conv_likelihood_CIFAR(7, x, w1_estimate, w2_estimate,w3_estimate, w4_estimate, w5_estimate,w6_estimate, part_w7,W8,W9,W10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights) # y_out shape =[batch,N,units]
        w7_estimate =  set_weights(part_w7, shape_w7, 6)
        part_w8=    BaSIS_Conv_likelihood_CIFAR(8, x, w1_estimate, w2_estimate,w3_estimate, w4_estimate, w5_estimate,w6_estimate, w7_estimate,part_w8,W9,W10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights) # y_out shape =[batch,N,units]
        w8_estimate =  set_weights(part_w8, shape_w8, 7)
        part_w9=    BaSIS_Conv_likelihood_CIFAR(9, x, w1_estimate, w2_estimate,w3_estimate, w4_estimate, w5_estimate,w6_estimate, w7_estimate,w8_estimate, part_w9,W10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights) # y_out shape =[batch,N,units]
        w9_estimate =  set_weights(part_w9, shape_w9, 8)
        part_w10=    BaSIS_Conv_likelihood_CIFAR(10, x, w1_estimate, w2_estimate,w3_estimate, w4_estimate, w5_estimate,w6_estimate, w7_estimate,w8_estimate,w9_estimate, part_w10,W11,  y,  kernel_num = num_kernels,kernel_size=kernels_size,N = N ,sigma_part= sigma_particles, weights = weights) # y_out shape =[batch,N,units]
        w10_estimate =  set_weights(part_w10, shape_w10, 9)

        part_w11, y_out  =  BaSIS_Flatten_and_FC_CIFAR( x, w1_estimate, w2_estimate,w3_estimate, w4_estimate, w5_estimate,w6_estimate, w7_estimate,w8_estimate,w9_estimate,w10_estimate,part_w11,  y_label=y, N = N ,sigma_part= sigma_particles,  weights = weights) # y_out shape =[batch,N,units]
        w11_estimate =  set_weights(part_w11, shape_w11, 10)    
        return  y_out, part_w1, part_w2, part_w3, part_w4, part_w5, part_w6, part_w7,part_w8, part_w9, part_w10, part_w11
    
    @tf.function
    def validation_on_batch(x, y, w1, w2, w3, w4, w5, w6,w7,w8,w9,w10,w11):                     
        mu_out, fc = cnn_model(x, training=False) 
        cnn_model.trainable = False        
        vloss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = fc, labels = y),axis=0)                                          
        regularization_loss=tf.math.add_n(cnn_model.losses)
        total_vloss = vloss + regularization_loss   
        particles_out, y_out, fc = CNN_model_test_CIFAR(x, w1, w2,w3, w4, w5, w6,w7,w8,w9,w10,w11,N= N, num_filters = num_kernels, kernel_size=kernels_size)
        return total_vloss, mu_out, particles_out, y_out
    @tf.function
    def test_on_batch(x, y, w1, w2, w3, w4, w5, w6,w7,w8,w9,w10,w11):  
        cnn_model.trainable = False                    
        mu_out, fc = cnn_model(x, training=False)  
        particles_out, y_out, fc = CNN_model_test_CIFAR(x, w1, w2, w3, w4, w5, w6,w7,w8,w9,w10,w11, num_filters = num_kernels, N=N, kernel_size=kernels_size)          
        return mu_out,particles_out, y_out , fc
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
        training_step =1
        adam1=0
        adam2=0
        first_step = True
        part_w1, part_w2, part_w3, part_w4, part_w5,part_w6,part_w7, part_w8, part_w9, part_w10, part_w11 = 0,0,0,0,0, 0, 0, 0, 0, 0, 0
        
        if continue_training:
            first_step = False # comment if training from a model with no particles
            saved_model_path = '/Cifar_10/epoch_{}/{}_particles/exp_ce/'.format(saved_model_epochs, N)
            cnn_model.load_weights(saved_model_path + 'pf_cnn_model')
            #comment next few lines to upload particles if NA
            init_1 = open(saved_model_path +'particles1.pkl', 'rb')
            part_w1, part_w2, part_w3, part_w4, part_w5, part_w6  = pickle.load(init_1)
            init_1.close()
            init_2 = open(saved_model_path +'particles2.pkl', 'rb')
            part_w7, part_w8,part_w9, part_w10, part_w11 = pickle.load(init_2)
            init_2.close()
            
            
        train_acc = np.zeros(epochs)
        train_acc2 = np.zeros(epochs)  
        valid_acc = np.zeros(epochs)
        valid_acc2 = np.zeros(epochs)
        train_err = np.zeros(epochs)
        valid_error = np.zeros(epochs)
        
        start = timeit.default_timer()       
        for epoch in range(epochs):
            print('Epoch: ', epoch+1, '/' , epochs)           
            acc1 = 0 #from model
            acc2 = 0 #from particles
            acc_valid1 = 0
            acc_valid2 = 0 
            err1 = 0
            err_valid1 = 0
            tr_no_steps = 0
            va_no_steps = 0           
            #-------------Training--------------------
            for step, (x, y) in enumerate(tr_dataset):                         
                update_progress(step/int(x_train.shape[0]/(batch_size)) )                
                
                gradients, loss,loss2, mu_out= get_gradient(x, y)
                if tr_no_steps == 0:
                    lr_current=lr 
                else:
                    lr_current = find_lr_polynomial(training_step)
                y_out, part_w1, part_w2,part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11 = transition_and_update(x, y, lr_current,  gradients,  part_w1, part_w2,part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11, first = first_step)           #y_out shape batch,N,units            
                first_step = False
                err1+= loss.numpy() 
                
                y_out = tf.reduce_mean(y_out, axis = 1)
                
                correct = tf.equal(tf.cast(tf.argmax(mu_out,-1),tf.int32),tf.cast(y,tf.int32))
                accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
                mu_out2, particles_out, y_out, fc   = test_on_batch(x, y, part_w1, part_w2,part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11 )
                accuracy2, particles_prediction = accuracy_test(particles_out, y)
                acc2+=accuracy2.numpy()
                acc1+=accuracy.numpy()

                if step % 50 == 0:
                    
                    print("\n Step:", step, "Loss:" , float(loss2.numpy()))
                    print("\n Step:", step, "Loss2:" , float(err1/(tr_no_steps + 1.)))
                    print("Total Training accuracy from model so far: %.3f" % float(acc1/(tr_no_steps + 1.)))     
                    print("Total Training accuracy from particles so far: %.3f" % float(acc2/(tr_no_steps + 1.)))                                                                
                tr_no_steps+=1 
                training_step+=1 
       
            train_acc[epoch] = acc1/tr_no_steps
            train_acc2[epoch] = acc2/tr_no_steps
            train_err[epoch] = err1/tr_no_steps        
            print('Training Acc  ', train_acc[epoch])
            print('Training error  ', train_err[epoch])        
                    
            #---------------Validation----------------------                  
            for step, (x, y) in enumerate(val_dataset):               
                update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
                
                total_vloss, mu_out, particles_out, y_out  = validation_on_batch(x, y, part_w1, part_w2 ,part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11 )                
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
            cnn_model.save_weights(PATH + 'pf_cnn_model')   

            print('Total Training Time: ', stop - start)
            print('Training Acc  ', train_acc[epoch], train_acc2[epoch])
            print('Validation Acc  ', valid_acc[epoch], valid_acc2[epoch])           
            print('------------------------------------')
            print('Training error  ', train_err[epoch])
            print('Validation error  ', valid_error[epoch])           
        #-----------------End Training--------------------------             
        cnn_model.save_weights(PATH + 'pf_cnn_model')        
        if (epochs > 1):
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_acc, 'b', label='Training acc')
            plt.plot(valid_acc,'r' , label='Validation acc')
            #plt.ylim(0, 1.1)
            plt.title("Density Propagation CNN on Fashion- MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend(loc='lower right')
            plt.savefig(PATH + 'VDP_CNN_on_FMNIST_Data_acc.png')
            plt.close(fig)    
    
            fig = plt.figure(figsize=(15,7))
            plt.plot(train_err, 'b', label='Training error')
            plt.plot(valid_error,'r' , label='Validation error')            
            plt.title("Density Propagation CNN on Fashion MNIST Data")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.legend(loc='upper right')
            plt.savefig(PATH + 'VDP_CNN_on_FMNIST_Data_error.png')
            plt.close(fig)
                         

        f1 = open(PATH + 'particles1.pkl', 'wb')         
        pickle.dump([part_w1.numpy(), part_w2.numpy(),part_w3.numpy(), part_w4.numpy(), part_w5.numpy(), part_w6.numpy() ], f1)                                                   
        f1.close()
        f2 = open(PATH + 'particles2.pkl', 'wb')         
        pickle.dump([part_w7.numpy(), part_w8.numpy(),part_w9.numpy(), part_w10.numpy(), part_w11.numpy() ], f2)                                                   
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
        textfile.write('\n sigma for particles all layers : ' +str(sigma_particles))
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
        print('testing on noise free')
        test_path = 'test_results/'
        cnn_model.load_weights(PATH + 'pf_cnn_model') 
        test_no_steps = 0        
        init_1 = open(PATH +'particles1.pkl', 'rb')
        part_w1, part_w2,part_w3, part_w4, part_w5, part_w6   = pickle.load(init_1)
        init_1.close()
        init_2 = open(PATH +'particles2.pkl', 'rb')
        part_w7, part_w8,part_w9, part_w10, part_w11   = pickle.load(init_2)
        init_2.close()
        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 3])
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
            mu_out, particles_out, y_out, fc   = test_on_batch(x, y, part_w1, part_w2,part_w3, part_w4, part_w5, part_w6,part_w7 , part_w8,part_w9, part_w10, part_w11)              
            stop = timeit.default_timer()
            mu_out_[test_no_steps,:,:] = y_out
            pf = particles_out.numpy()  
            all_particles [test_no_steps,:,:,: ] = pf
            pf_fc = fc.numpy()   
            v = np.var(pf, axis=0)  
            v_fc  = np.var(pf_fc, axis = 0)  
            skew = scipy.stats.skew(pf, axis=0)
            kurt = scipy.stats.kurtosis(pf, axis=0)
            
            var_out[test_no_steps,:,:] = v
            var_out_fc[test_no_steps,:,:] = v_fc
               
            correct = tf.equal(tf.cast(tf.argmax(mu_out,-1),tf.int32),tf.cast(y,tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
            
            #y=tf.cast(y,tf.int64) 
            accuracy2, particles_prediction = accuracy_test(particles_out, y, MAP=False)
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
        test_no_steps = 0    
        test_path = 'test_results_random_noise_{}/'.format(gaussian_noise_var)
        cnn_model.load_weights(PATH + 'pf_cnn_model')       
        init_1 = open(PATH +'particles1.pkl', 'rb')
        part_w1, part_w2,part_w3, part_w4, part_w5, part_w6   = pickle.load(init_1)
        init_1.close()
        init_2 = open(PATH +'particles2.pkl', 'rb')
        part_w7, part_w8,part_w9, part_w10, part_w11   = pickle.load(init_2)
        init_2.close()
        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 3])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])

        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        var_out = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
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
                noise = tf.random.normal(shape = [batch_size, input_dim, input_dim, 3], mean = 0.0, stddev = gaussain_noise_std, dtype = x.dtype , seed=13) 
                x = x +  noise
                x = tf.clip_by_value(x, 0.0,1.0)
            true_x[test_no_steps, :, :, :,:] = x
            mu_out, particles_out, y_out,fc   = test_on_batch(x, y, part_w1, part_w2, part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11 )              
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
            
            correct = tf.equal(tf.cast(tf.argmax(mu_out,-1),tf.int32),tf.cast(y,tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
            
            y=tf.cast(y,tf.int64)
            
            accuracy2, particles_prediction = accuracy_test(particles_out, y)
            acc_test[test_no_steps] = accuracy.numpy()
            acc_test2[test_no_steps] = accuracy2.numpy()
            y =y.numpy()
            for j in range(batch_size):
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
        
        p = open(PATH + test_path + 'variance_info.pkl', 'wb')      
        pickle.dump([variance_correct, var, variance_incorrect], p)                                                  
        p.close()

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
        textfile.close()
        textfile = open(PATH + test_path + 'Results_for_excel.txt','w')   
        if Random_noise:
            textfile.write('\n Random Noise std: '+ str(gaussian_noise_var))   
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
    
    
    elif(Adversarial_noise):
        init_1 = open(PATH +'particles1.pkl', 'rb')
        part_w1, part_w2,part_w3, part_w4, part_w5, part_w6   = pickle.load(init_1)
        init_1.close()
        init_2 = open(PATH +'particles2.pkl', 'rb')
        part_w7, part_w8,part_w9, part_w10, part_w11   = pickle.load(init_2)
        init_2.close()
        
        
        if Targeted:
            test_path = 'test_results_targeted_adversarial_noise_{}/'.format(epsilon)            
        else:
            test_path = 'test_results_non_targeted_adversarial_noise_{}/'.format(epsilon)              
        cnn_model.load_weights(PATH + 'pf_cnn_model')       
        test_no_steps = 0        
        

        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 3])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 3])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        var = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
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
            mu_out, particles_out, y_out,fc   = test_on_batch(adv_x, y,  part_w1, part_w2, part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11 )           
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
                snr_signal[step,j] = 10*np.log10( np.sum(np.square(x[j,:,:, :]))/np.sum( np.square(epsilon*adv_perturbations[test_no_steps, j, :, :, :]  ) ))
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
        
        
        print('Output Variance', np.mean(var))
        print('SNR', np.mean(snr_signal))     

        p = open(PATH + test_path + 'variance_info.pkl', 'wb')      
        pickle.dump([variance_correct, var, variance_incorrect], p)                                                  
        p.close()  
        
        
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
       
        textfile.write( str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write( str(np.mean(np.absolute(kurt_correct)))  )
        
        textfile.write( str(np.mean(np.absolute(kurt_incorrect)))  )
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
    

    elif(PGD_Adversarial_noise):
        init_1 = open(PATH +'particles1.pkl', 'rb')
        part_w1, part_w2,part_w3, part_w4, part_w5, part_w6   = pickle.load(init_1)
        init_1.close()
        init_2 = open(PATH +'particles2.pkl', 'rb')
        part_w7, part_w8,part_w9, part_w10, part_w11   = pickle.load(init_2)
        init_2.close()
        
        if Targeted:
            test_path = 'test_results_targeted_PGDadversarial_noise_{}_max_iter_{}_{}/'.format(epsilon, maxAdvStep, stepSize)
        else:
            test_path = 'test_results_non_targeted_PGDadversarial_noise_{}_max_iter_{}_{}/'.format(epsilon, maxAdvStep, stepSize)
        cnn_model.load_weights(PATH + 'pf_cnn_model')       
        test_no_steps = 0        
        
        true_x = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 3])
        adv_perturbations = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, input_dim, input_dim, 3])
        true_y = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size])
        mu_out_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        sigma_ = np.zeros([int(x_test.shape[0] / (batch_size)), batch_size, class_num])
        acc_test = np.zeros(int(x_test.shape[0] / (batch_size)))
        var = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        snr_signal = np.zeros([int(x_test.shape[0] /batch_size) ,batch_size])
        third_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        fourth_moment = np.zeros([int(x_test.shape[0] / (batch_size)) ,batch_size])
        all_particles = np.zeros([int(x_test.shape[0] / (batch_size)),N, batch_size, class_num])
        
        variance_correct = []
        variance_incorrect = []
        skew_correct = []
        skew_incorrect = []
        kurt_correct = []
        kurt_incorrect = []
        epsilon  = epsilon
        for step, (x, y) in enumerate(val_dataset):
            update_progress(step / int(x_test.shape[0] / (batch_size)) ) 
            
            true_y[test_no_steps, :] = y
            adv_x = x + tf.random.uniform(x.shape, minval=-epsilon, maxval=epsilon)
            adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)
            for advStep in range(maxAdvStep):
                if Targeted:
                    y_hot = tf.one_hot(y, depth=class_num)
                    y_true_batch = np.zeros_like(y_hot)
                    y_true_batch[:, adversary_target_cls] = 1.0
                    adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(adv_x, y_true_batch)
                else:
                    adv_perturbations[test_no_steps, :, :, :] = create_adversarial_pattern(adv_x, y)
                adv_x = adv_x + stepSize * adv_perturbations[test_no_steps, :, :, :]
                adv_x = tf.clip_by_value(adv_x, x - epsilon, x + epsilon)
                adv_x = tf.clip_by_value(adv_x, 0.0, 1.0)                          
            
            true_x[test_no_steps, :, :, :,:] = adv_x
            mu_out, particles_out, y_out,fc   = test_on_batch(adv_x, y,  part_w1, part_w2, part_w3, part_w4, part_w5, part_w6,part_w7, part_w8,part_w9, part_w10, part_w11 )           
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
                snr_signal[step,j] = 10*np.log10( np.sum(np.square(x[j,:,:, :]))/np.sum( np.square(epsilon*adv_perturbations[test_no_steps, j, :, :, :]  ) ))
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

        if PGD_Adversarial_noise:
            if Targeted:
                textfile.write('\n Adversarial attack: TARGETED')
                textfile.write('\n The targeted attack class: ' + str(adversary_target_cls))                   
            else:      
                textfile.write('\n Adversarial attack: Non-TARGETED')
            textfile.write('\n Adversarial Noise epsilon: '+ str(epsilon ))    
            textfile.write("\n SNR: "+ str(np.mean(snr_signal)))               
            textfile.write("\n---------------------------------")  
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write('\n Adversarial Noise epsilon: ' + str(epsilon))
            textfile.write("\n stepSize: "+ str(stepSize)) 
            textfile.write("\n Maximum number of iterations: "+ str(maxAdvStep))
          
        textfile.close() 
        textfile = open(PATH + test_path + 'Results_for_excel.txt','w')   
        if PGD_Adversarial_noise:
              
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
       
        textfile.write( str(np.mean(np.absolute(fourth_moment)))) 
        textfile.write( str(np.mean(np.absolute(kurt_correct)))  )
        
        textfile.write( str(np.mean(np.absolute(kurt_incorrect)))  )
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
    
    
if __name__ == '__main__':
    main_function(Training=False,  Random_noise=True, gaussain_epsilon=0.2)
    