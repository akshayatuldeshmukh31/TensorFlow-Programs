import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import math
import pandas as pd
import numpy as np

training_set_df = pd.read_csv('creditcard_training_set.csv')
training_set_df = training_set_df.sample(frac=1)
training_set_df = training_set_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'Fraud', 'Normal']]
training_set_np = training_set_df.as_matrix()
training_set_size = len(training_set_np)

test_set_df = pd.read_csv('creditcard_test_set.csv')
test_set_df = test_set_df.sample(frac=1)
test_set_df = test_set_df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'Fraud', 'Normal']]
test_set_size = len(test_set_df.index)
test_set_np = test_set_df.as_matrix()

num_nodes = 52
num_features = 27

num_node_layer_1 = num_nodes
num_node_layer_2 = math.ceil(num_node_layer_1 * 1.5)
num_node_output_layer = 2

X = tf.placeholder('float', [None, num_features])
Y = tf.placeholder('float', [None, 2])
global_step = tf.Variable(0, trainable=False)
keep_prob = tf.placeholder('float')

hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([num_features, num_node_layer_1])), 'biases': tf.Variable(tf.random_normal([num_node_layer_1]))}
hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([num_node_layer_1, num_node_layer_2])), 'biases': tf.Variable(tf.random_normal([num_node_layer_2]))}
output_layer = {'weights': tf.Variable(tf.random_normal([num_node_layer_2, num_node_output_layer])), 'biases':tf.Variable(tf.random_normal([num_node_output_layer]))}

def nn_model(data):
	
	layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	layer_1 = tf.nn.relu(layer_1)

	layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	layer_2 = tf.nn.relu(layer_2)
	layer_2 = tf.nn.dropout(layer_2, keep_prob)

	output = tf.add(tf.matmul(layer_2, output_layer['weights']), output_layer['biases'])

	return output

def train_nn(data):

	prediction = nn_model(data)
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
	learning_rate = tf.train.exponential_decay(0.001, global_step, 10, 0.96, staircase=False)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	num_epochs = 10000

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0

			epoch_X = training_set_np[:,0:27]
			epoch_Y = training_set_np[:,27:29]
			op, c = sess.run([optimizer, cost], feed_dict={X: epoch_X, Y: epoch_Y, keep_prob: 0.75})
			epoch_loss += c

			print("Epoch", str(epoch+1), 'out of', num_epochs, ': ', epoch_loss)

		correctness = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

		test_X = test_set_np[:,0:27]
		test_Y = test_set_np[:,27:29]
		print("Accuracy:", accuracy.eval({X: test_X, Y: test_Y, keep_prob: 1}))

train_nn(X)