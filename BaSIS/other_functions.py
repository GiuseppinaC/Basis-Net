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
############################################################################
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
##########################################################################
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def create_heatmap(path_full, u_original):
        pf = open(path_full + 'all_variance.pkl', 'wb')                   
        pickle.dump([u_original], pf)                                                  
        pf.close()

        u_original = NormalizeData(u_original)
        u = np.mean(u_original, axis = 0)

        plt.figure(figsize=(10,10))
        im= plt.imshow(u, cmap='winter_r', interpolation='nearest')
        plt.title("Uncertainty map") 
        add_colorbar(im)
        plt.clim(0,1)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig(path_full +'uncertainty_heatmap.png')
        plt.close()
        


###################################### function for histogram of incorrect
# mostly for adversarial
def histogram_adv(folder,n_images, numbers_wrong, x, y_test,mean_p,  variance, skew, kurt,pf, in_seed=1 ):
    new_folder = '/histograms_adv/'
    
    if not os.path.exists(folder+ new_folder ):
            print('creating the folder')
            os.makedirs(folder+ new_folder )
    random.seed(in_seed)
    numbers=random.sample(numbers_wrong,n_images)
    for j in range(n_images):
        n=numbers[j]
        y=y_test[n]
        lab=mean_p[n]
        image = x[n,:,:]
        v2 = variance[n]
        s2 = skew[n]
        k2 = kurt[n]
        print('observation', n)
        v,s,k = round(v2,5),round(s2,3),round(k2,3)
        print('skew ', s, 'kurt ',k, 'var' ,v)
        plt.figure(figsize=(8,6))
        for i in range(10):
   
            plt.hist(pf[n,:,i], bins=100, alpha=0.5, label="label_{}".format(i))
        plt.xlabel("Softmax Score", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Prediction {} - Var: {}, Skew: {}, Kurt: {}".format(lab,v,s,k ))
        plt.xlim([0, 1])
        plt.legend(loc='best')
        plt.savefig('{}/{}/Histogram_observation_{}_true_{}.png'.format(folder,new_folder, n,y))
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.hist(pf[n,:,lab], bins=100, alpha=0.5, label="label_{}".format(i))
        plt.xlabel("Softmax Score", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Prediction {} - Var: {}, Skew: {}, Kurt: {}".format(lab,v,s,k ))
        plt.xlim([0, 1])
        #plt.legend(loc='upper right')
        plt.savefig('{}/{}/Histogram_observation_{}_true_{}_only_prediction.png'.format(folder,new_folder, n,y))
        plt.clf()

################# function to generate the prediction histograms ##########
def histogram_with_label(folder,n_images, in_seed, x, y_test,mean_p, variance, skew, kurt,pf ):
    new_folder = '/histograms/'
    folder_seed = 'seed_{}/'.format(in_seed)
    random.seed(in_seed)
    if not os.path.exists(folder+ new_folder+folder_seed ):
            print('creating the folder')
            os.makedirs(folder+ new_folder+folder_seed )
    sequence = [i for i in range(10000)]
    numbers=random.sample(sequence,n_images)
    for j in range(n_images):
        n=numbers[j]
        y=y_test[n]
        lab=mean_p[n]
        image = x[n,:,:]
        v2 = variance[n]
        s2 = skew[n]
        k2 = kurt[n]
        print('observation', n)
        v,s,k = round(v2,5),round(s2,3),round(k2,3)
        print('skew ', s, 'kurt ',k, 'var' ,v)
        plt.figure(figsize=(8,6))
        for i in range(10):
   
            plt.hist(pf[n,:,i], bins=100, alpha=0.5, label="label_{}".format(i))
        plt.xlabel("Softmax Score", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Prediction {} - Var: {}, Skew: {}, Kurt: {}".format(lab,v,s,k ))
        plt.xlim([0, 1])
        plt.legend(loc='best')
        plt.savefig('{}/{}/{}/Histogram_observation_{}_true_{}.png'.format(folder,new_folder,folder_seed, n,y))
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.hist(pf[n,:,lab], bins=100, alpha=0.5, label="label_{}".format(i))
        plt.xlabel("Softmax Score", size=14)
        plt.ylabel("Count", size=14)
        plt.title("Prediction {} - Var: {}, Skew: {}, Kurt: {}".format(lab,v,s,k ))
        plt.xlim([0, 1])
        plt.savefig('{}/{}/{}/Histogram_observation_{}_true_{}_only_prediction.png'.format(folder,new_folder,folder_seed, n,y))
        plt.clf()
        

    fig, ax = plt.subplots(2,int(n_images/2))
    
    for j, ax in enumerate(ax.flatten()):
        n=numbers[j]
        image = x[n,:,:]
        lab=mean_p[n]
        ax.imshow(image, cmap='gray_r')
        ax.set_title('Observation: {} \n truth: {}'.format(n, lab))
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig('{}/{}/{}/Inputs_images.png'.format(folder,new_folder,folder_seed))
    plt.title('truth: {}'.format(lab))
    plt.clf()
    print('creating an excel file')

    Norm = pd.ExcelWriter(folder+ 'Moments_info.xlsx', engine='xlsxwriter')
    df1 = pd.DataFrame(y_test, columns={'true label'})
    df1.to_excel(Norm, 'Sheet', startrow=0, startcol=0)
    df2 = pd.DataFrame(mean_p, columns={'prediction'})
    df2.to_excel(Norm, 'Sheet', startrow=0, startcol=4)
    df1 = pd.DataFrame(variance, columns = {'var'})
    df1.to_excel(Norm, 'Sheet', startrow=0, startcol=8)
    df1 = pd.DataFrame(skew, columns = {'skew'})
    df1.to_excel(Norm, 'Sheet', startrow=0, startcol=12)
    df1 = pd.DataFrame(kurt, columns = {'kurt'})
    df1.to_excel(Norm, 'Sheet', startrow=0, startcol=16)
    Norm.save()
    print('done saving excel file')
#################################################################################################
def accuracy_test2(particles, labels):
    """ args: particles shape=[N, batch_size, num_labels] 
              labels shape=[batch_size,]"""
    
    pred = tf.reduce_mean(particles, axis = 0) 
    argmax = tf.argmax(pred,-1)
    correct = tf.equal(tf.cast(argmax,tf.int32),tf.cast(labels,tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
    return accuracy 
  
def accuracy_test(s, labels, MAP=False):
    """ args: particles shape=[N, batch_size, num_labels] 
              labels shape=[batch_size,]
              MAP:  maximum a posteriori probability estimate"""
    if MAP:
        "input s = particles; shape = [N, batch, num_labels]"
        pred = np.zeros([s.shape[1], s.shape[2]])
        for i in range(s.shape[1]): # loop over batch
         for j in range(s.shape[2]):  # loop over labels
            counts, bins = np.histogram(s[:,i,j], bins=1000)
            max_bin = np.argmax(counts)
            x = bins[max_bin:max_bin+2].mean()
    
            pred[i,j] = x
    else:
        pred = tf.reduce_mean(s, axis = 0) 
    argmax = tf.argmax(pred,-1)
    correct = tf.equal(tf.cast(argmax,tf.int32),tf.cast(labels,tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32)) 
    return accuracy , argmax


def hypothesis_test1(x_1, x_2):
    s1 = tf.math.reduce_variance(x_1)
    s2 = tf.math.reduce_variance(x_2)
    n1 = len(x_1)
    n2 = len(x_2)
    mu_x1  = tf.math.reduce_mean(x_1)
    mu_x2  = tf.math.reduce_mean(x_2)
    denominator = tf.math.sqrt((s1/n1)+(s2/n2))
    test_statistic = ((mu_x1-mu_x2)/denominator)
    pval = 2*(norm.sf(abs(test_statistic)))
    return np.round(test_statistic, 3), np.round(pval, 4)

def hypothesis_test(x_1, x_2):
    
    ztest ,pval = scipy.stats.ttest_ind(x_1,x_2, equal_var=False)
    print('pval', float(pval))
    if pval<0.05:
        print("reject null hypothesis")
    else:
        print("accept null hypothesis")
    return ztest, pval