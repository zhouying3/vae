#https://www.leiphone.com/news/201704/PgRxGpwtFpSgJoAZ.html
import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,n_layer,act_fun=None):
	layer_name = "layer%s"%n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]))
			tf.histogram_summary(layer_name+'weights',Weights)
		with tf.name_scope('bias'):
			bias = tf.Variable(tf.zeros([out_size])+0.1)
			tf.histogram_summary(layer_name+'bias',bias)
		with tf.name_scope('activation'):
			act = tf.matmul(inputs,Weights)+bias
			tf.histogram_summary(layer_name+'activation',act)
			if act_fun != None:
				outputs = act_fun(act)
			else:
				outputs = act
			tf.histogram_summary(layer_name+'outputs',outputs)
	return outputs
	
with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32,[None,1],name='x_input')
	y = tf.placeholder(tf.float32,[None,1],name='y_input')
	
l1 = add_layer(x,1,10,n_layer=1,act_fun=tf.nn.relu)
prediction = add_layer(l1,10,1,n_layer=2,act_fun=None)

with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(
	tf.square(y-prediction),reduction_indices=[1]))
	tf.scalar_summary('loss',loss)
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.001).minimizer(loss)
	
init =tf.initialize_all_variables()
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter('Desktop/'.sess.garph)
sess.run(init)

x_data = np.linespace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data)-0.5+noise

for step in range(1000):
	sess.run(train_step,feed_dict={x:x_data,y:y_data})
	if step %50==0:
		result = sess.run(merged,feed_dict={x:x_data,y:y_data})
		writer.add_summary(result)