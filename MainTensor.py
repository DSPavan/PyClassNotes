# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 09:54:52 2019

@author: pavan
"""

import tensorflow as tf 
import matplotlib.pyplot as plt
import os
import time
from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

sess = tf.Session()
a = tf.truncated_normal([16,128,128,3])
sess.run(tf.global_variables_initializer())
sess.run(tf.shape(a))

b=tf.reshape(a,[16,128*128*3])
sess.run(tf.shape(b))
print(a)


os.system("python C:\\users\\pavan\\dataset.py")
#print("loaded py file")
import dataset as dst
#print('ld Dataset')

#########
batch_size = 32

#Prepare input data
classes1 = os.listdir('C:\\Users\\pavan\\Downloads\\evaluation')
classes = ['dog', 'cat']
num_classes = len(classes)
print(num_classes)
print(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path='C:\\Users\\pavan\\Downloads\\evaluation'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dst.read_train_sets(train_path, img_size, classes1, validation_size=validation_size)
print(data)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))

#tf.summary.FileWriterCache.clear()
session = tf.Session()
logdir = 'C:\\Users\\pavan\\Downloads\\dt2'
writer = tf.summary.FileWriter(logdir)  # create writer
writer.add_graph(session.graph)

x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 128

layer_conv1 = dst.create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = dst.create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= dst.create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = dst.create_flatten_layer(layer_conv3)

layer_fc1 = dst.create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = dst.create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saverx = tf.train.Saver()

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            #s=session.run(merged_summary,feed_dict={x: x_batch,   y_true: y_true_batch})
           # write.add_summary(s,i)
                 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saverx.save(session, 'C:\\Users\\pavan\\Downloads\\dt2\\dogs-cats-model') 


    total_iterations += num_iteration


session.run(tf.global_variables_initializer()) 

tf.summary.scalar("cross-entropy", cross_entropy)
tf.summary.scalar("accuracy",accuracy)


train(num_iteration=100)















