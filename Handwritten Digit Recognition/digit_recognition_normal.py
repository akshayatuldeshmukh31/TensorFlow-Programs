import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

hidden_layer_1_num_nodes = 500
hidden_layer_2_num_nodes = 500
hidden_layer_3_num_nodes = 500

num_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
global_step = tf.Variable(0, trainable=False)
keep_prob = tf.placeholder('float')

hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784, hidden_layer_1_num_nodes])), 'biases':tf.Variable(tf.random_normal([hidden_layer_1_num_nodes]))}
hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([hidden_layer_1_num_nodes, hidden_layer_2_num_nodes])), 'biases':tf.Variable(tf.random_normal([hidden_layer_2_num_nodes]))}
hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([hidden_layer_2_num_nodes, hidden_layer_3_num_nodes])), 'biases':tf.Variable(tf.random_normal([hidden_layer_3_num_nodes]))}
output_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer_3_num_nodes, num_classes])), 'biases':tf.Variable(tf.random_normal([num_classes]))}

def neural_network_model(data):

	layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1['biases'])
	layer_1 = tf.nn.relu(layer_1)
	
	layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	layer_2 = tf.nn.relu(layer_2)
	layer_2 = tf.nn.dropout(layer_2, keep_prob)
	
	layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	layer_3 = tf.nn.relu(layer_3)
	layer_3 = tf.nn.dropout(layer_3, keep_prob)
	
	output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']

	return output

def train_nn(data):

	prediction = neural_network_model(data)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	learning_rate = tf.train.exponential_decay(0.001, global_step, 2000, 0.96, staircase=True)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step = global_step)

	num_epochs = 100

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0

			for i in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.75})
				epoch_loss += c

			print('Epoch', epoch+1, 'out of ', num_epochs, ': ', epoch_loss)

		correctness = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

		print("Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

train_nn(x)