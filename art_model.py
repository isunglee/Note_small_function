
# coding: utf-8

# In[ ]:

import img_group
import tensorflow as tf
import numpy as np

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    
    # l1a shape = (?, 100, 100, 32)
    # l1 shpae = (?, 50, 50, 32)
    biases = tf.get_variable('biases1', [32], initializer=tf.constant_initializer(0.0))
    conv1 = tf.nn.conv2d(X, w, strides=[1,1,1,1], padding='SAME')
    bias = tf.nn.bias_add(conv1, biases)
    l1a = tf.nn.relu(bias)
    l1 = tf.nn.max_pool(l1a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    
    # l2a shape = (?, 50, 50, 64)
    # l2 shpae = (?, 25, 25, 64)
    biases = tf.get_variable('biases2', [64], initializer=tf.constant_initializer(0.1))
    conv2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME')
    bias = tf.nn.bias_add(conv2, biases)
    l2a = tf.nn.relu(bias)
    l2 = tf.nn.max_pool(l2a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    
    # l3a shape = (?, 25, 25, 128)
    # l3 shpae = (?, 13, 13, 128)
    # l3 reshape to (?, 128*13*13)
    biases = tf.get_variable('biases3', [128], initializer=tf.constant_initializer(0.1))
    conv3 = tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME')
    bias = tf.nn.bias_add(conv3, biases)
    l3a = tf.nn.relu(bias)
    l3 = tf.nn.max_pool(l3a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)
    
    biases = tf.get_variable('biases4', [625], initializer=tf.constant_initializer(0.1))
    l4 = tf.nn.relu(tf.matmul(l3, w4) + biases)
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    
    biases = tf.get_variable('biases5', [23], initializer=tf.constant_initializer(0.0))
    pyx = tf.matmul(l4, w_o) + biases
    return pyx

result = img_group.get_data()
trX = np.array(result['train_images'])
trY = np.array(result['train_labels'])
teX = np.array(result['test_images'])
teY = np.array(result['test_labels'])

trX = trX.reshape(-1, 100, 100, 3)
teX = teX.reshape(-1, 100, 100, 3)

X = tf.placeholder('float', [None, 100, 100, 3])    # img shape = (100, 100, 3)
Y = tf.placeholder('float', [None, 23])             # label = 23

w = init_weights([5, 5, 3, 32])
w2 = init_weights([5, 5, 32, 64])
w3 = init_weights([5, 5, 64, 128])
w4 = init_weights([128 * 13 * 13, 625])
w_o = init_weights([625, 23])

p_keep_conv = tf.placeholder('float')
p_keep_hidden = tf.placeholder('float')

py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.009, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    
    for i in xrange(50):
        training_batch = zip(xrange(0, len(trX), batch_size), xrange(batch_size, len(trX), batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X:trX[start:end], Y:trY[start:end], p_keep_conv:0.8, p_keep_hidden:0.5})
        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        print (i, np.mean(np.argmax(teY[test_indices], axis=1) == 
                          sess.run(predict_op, feed_dict={X:teX[test_indices],
                                                          p_keep_conv:1.0,
                                                          p_keep_hidden:1.0})))

