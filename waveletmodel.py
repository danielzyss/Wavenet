from hyperparameters import *
from parameters import *
from tools import cwt

def Wavenet(input_data):

    transform1 = cwt(input_data, Level_Of_Decomposition_First_Layer)

    subtract1 = tf.subtract(tf.abs(transform1), th_value_L1)
    relu1 = tf.nn.relu(subtract1)
    relu1 = tf.multiply(relu1, tf.sign(transform1))

    relu1 = tf.expand_dims(relu1, 3)

    max_pool1 = tf.nn.max_pool(relu1, ksize=[1, maxpool_size_L1, maxpool_size_L1, 1],
                               strides=[1, maxpool_size_L1, maxpool_size_L1, 1], padding='SAME')

    max_pool1 = tf.reshape(max_pool1, shape=[int(max_pool1.get_shape().as_list()[0]), 1, int(Level_Of_Decomposition_First_Layer/maxpool_size_L1 *
                                                                TS_length/(maxpool_size_L1))])

    transform2 = cwt(max_pool1, Level_Of_Decomposition_Second_Layer)

    subtract2 = tf.subtract(tf.abs(transform2), th_value_L2)
    relu2 = tf.nn.relu(subtract2)
    relu2 = tf.multiply(relu2, tf.sign(transform2))

    relu2 = tf.expand_dims(relu2, 3)


    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, maxpool_size_L2, maxpool_size_L2, 1],
                               strides=[1, maxpool_size_L2, maxpool_size_L2, 1], padding='SAME')


    new_shape = [max_pool2.get_shape().as_list()[0], full_shape_L3[0]]
    flat_output = tf.reshape(max_pool2, shape=new_shape)

    fully_connected1 = tf.nn.sigmoid(tf.add(tf.matmul(flat_output, full_weight_L3), full_bias_L3))
    final_model_output = tf.add(tf.matmul(fully_connected1, full_weight_L4), full_bias_L4)

    return (final_model_output)