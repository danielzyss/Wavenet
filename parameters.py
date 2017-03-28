import tensorflow as tf
from hyperparameters import *

input_shape = [batch_size, 1, TS_length]
X_input = tf.placeholder(dtype=tf.float32, shape=input_shape)
Y_target = tf.placeholder(dtype=tf.int32, shape=[batch_size])

eval_input_shape = [evaluation_size, 1, TS_length]
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.float32, shape=(evaluation_size))

th_shape_L1 = [1, 1, TS_length]

th_value_L1 = tf.Variable(tf.truncated_normal(shape=th_shape_L1, mean=0.0, stddev=0.5))

th_shape_L2 = [1, 1, int(TS_length/maxpool_size_L1)]

th_value_L2 = tf.Variable(tf.truncated_normal(shape=th_shape_L2, mean=0.0, stddev=0.5))

full_shape_L3 = [int((Level_Of_Decomposition_First_Layer/maxpool_size_L1)*Level_Of_Decomposition_Second_Layer/maxpool_size_L2 *
                     TS_length/(maxpool_size_L1*maxpool_size_L2)),
                 Size_Third_Layer]

full_weight_L3 = tf.Variable(tf.truncated_normal(shape=full_shape_L3, stddev=0.1 ,dtype=tf.float32))
full_bias_L3 = tf.Variable(tf.truncated_normal(shape=[full_shape_L3[1]], stddev=0.1, dtype=tf.float32))

full_weight_L4 = tf.Variable(tf.truncated_normal([full_shape_L3[1], target_size], stddev=0.1, dtype=tf.float32))
full_bias_L4 = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))