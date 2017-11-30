import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

num_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
global_step = tf.Variable(0, trainable=False)
keep_prob = tf.placeholder('float')

weights = {'layer1_conv_weights': tf.Variable(tf.random_normal([3, 3, 1, 32]))
			, 'layer2_conv_weights': tf.Variable(tf.random_normal([3, 3, 32, 64]))
			, 'fully_connected_layer3_weights': tf.Variable(tf.random_normal([7*7*64, 1024]))
			, 'output_layer_weights': tf.Variable(tf.random_normal([1024, num_classes]))}

biases = {'layer1_conv_biases': tf.Variable(tf.random_normal([32]))
			, 'layer2_conv_biases': tf.Variable(tf.random_normal([64]))
			, 'fully_connected_layer3_biases': tf.Variable(tf.random_normal([1024]))
			, 'output_layer_biases': tf.Variable(tf.random_normal([num_classes]))}

def neural_network_model(data):

	data = tf.reshape(data, shape=[-1, 28, 28, 1])

	layer1_conv = tf.nn.relu(tf.nn.conv2d(data, weights['layer1_conv_weights'], strides=[1, 1, 1, 1], padding='SAME') + biases['layer1_conv_biases'])
	layer1_conv = tf.nn.max_pool(layer1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	layer2_conv = tf.nn.relu(tf.nn.conv2d(layer1_conv, weights['layer2_conv_weights'], strides=[1, 1, 1, 1], padding='SAME') + biases['layer2_conv_biases'])
	layer2_conv = tf.nn.max_pool(layer2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	layer2_conv = tf.reshape(layer2_conv, shape=[-1, 7*7*64])

	fully_connected_layer3 = tf.nn.relu(tf.matmul(layer2_conv, weights['fully_connected_layer3_weights']) + biases['fully_connected_layer3_biases'])
	fully_connected_layer3 = tf.nn.dropout(fully_connected_layer3, keep_prob)

	output_layer = tf.matmul(fully_connected_layer3, weights['output_layer_weights']) + biases['output_layer_biases']

	return output_layer

def train_nn(data):

	prediction = neural_network_model(data)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	learning_rate = tf.train.exponential_decay(0.001, global_step, 2000, 0.96, staircase=True)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)

	num_epochs = 50

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0

			for i in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.75})
				epoch_loss += c

			print('Epoch', epoch+1, 'out of', num_epochs, ': ', epoch_loss)

		correctness = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

		print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

train_nn(x)