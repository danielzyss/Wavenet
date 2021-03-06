import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal as sgl
from hyperparameters import *

def loadDataFrames(l):
    ADeyesclosed = pd.read_pickle('data/ADeyesclosed.pk').as_matrix()
    ADeyesopened = pd.read_pickle('data/ADeyesopened.pk').as_matrix()

    EyesClosedMatrix = []
    EyesOpenedMatrix = []
    j = 29

    for n in range(int(len(ADeyesclosed[0, 0]) / l)):
        for i in range(ADeyesclosed.shape[0]):
            EyesClosedMatrix.append(np.array(ADeyesclosed[i, j][n * l: n * l + l]))
        for i in range(ADeyesopened.shape[0]):
            EyesOpenedMatrix.append(np.array(ADeyesopened[i, j][n * l:n * l + l]))

    EyesClosedMatrix = np.array(EyesClosedMatrix)
    EyesOpenedMatrix = np.array(EyesOpenedMatrix)

    ADcompletetrain = np.concatenate((EyesOpenedMatrix[:int(train_set_size / 2)],
                                      EyesClosedMatrix[:int(train_set_size / 2)]), axis=0)

    ADcompletetrain = matrixnormalise(ADcompletetrain, 0, 1)
    ADcompletetrain = np.expand_dims(ADcompletetrain, 1)

    ADcompletetest = np.concatenate(
        (EyesOpenedMatrix[int(train_set_size / 2):int(train_set_size / 2) + int(test_set_size / 2)],
         EyesClosedMatrix[int(train_set_size / 2):int(train_set_size / 2) + int(test_set_size / 2)]), axis=0)
    ADcompletetest = matrixnormalise(ADcompletetest, 0, 1)
    ADcompletetest = np.expand_dims(ADcompletetest, 1)

    y_train = np.concatenate(
        (np.zeros((int(train_set_size / 2),), dtype=np.int32), np.ones((int(train_set_size / 2)), dtype=int)), axis=0)
    y_test = np.concatenate(
        (np.zeros((int(test_set_size / 2),), dtype=np.int32), np.ones((int(test_set_size / 2),), dtype=int)), axis=0)

    return ADcompletetrain, y_train, ADcompletetest, y_test

def cwt(input_data, widthCwt):

    total_batch = input_data.get_shape().as_list()[0]
    nbwave = input_data.get_shape().as_list()[1]
    length = input_data.get_shape().as_list()[2]

    output_data = []
    for batch in range(total_batch):
        wavresult = []
        for w in range(nbwave):
            wav = input_data[batch, w]
            wav = tf.expand_dims(wav, 0)
            wav = tf.expand_dims(wav, 2)
            wav = tf.expand_dims(wav, 3)

            # While loop functions
            def body(i, m):
                v = conv1DWavelet(wav, i, rickerWavelet)
                v = tf.expand_dims(v, 1)
                m = tf.concat(1, [m,v])
                return [1 + i, m]

            def cond_(i, m):
                return tf.less_equal(i, widthCwt)

            # Initialize and run while loop
            emptyCwtMatrix = tf.zeros([length, 0], dtype='float32')
            i = tf.constant(1)
            _, result = tf.while_loop(cond_, body, [i, emptyCwtMatrix], back_prop=False, parallel_iterations=1024,
                                      shape_invariants=[i.get_shape(), tf.TensorShape([length, None])])
            result = tf.transpose(result)

            wavresult.append(result)
        wavresult = tf.stack(wavresult)
        output_data.append(wavresult)

    output_data = tf.stack(output_data, 0)
    output_data = tf.reshape(output_data, shape=[total_batch, int(widthCwt * nbwave), length])

    return output_data

def icwt(input_data, widthCwt):

    total_batch = input_data.get_shape()[0]
    len_TS = input_data.get_shape()[2]
    output_data = []

    for batch in range(total_batch):
        Xbatch = input_data[batch]

        i = tf.constant(0, dtype=tf.int32)

        def body(X_ARR, Xbatch, i):
            squareroot = tf.sqrt(tf.to_float(i))
            new_wave = tf.divide(Xbatch[tf.to_int32(i)],squareroot)
            new_wave = tf.expand_dims(new_wave, 0)
            i = tf.add(i, tf.constant(1))

            if i==1:
                X_ARR = new_wave

            else:
                X_ARR = tf.concat(0, [X_ARR, new_wave])

            return [X_ARR, Xbatch, i]

        def condition(X_ARR, Xbatch, i):
            return tf.not_equal(i, widthCwt)

        X_ARR = tf.zeros(shape=[1,Xbatch[0].get_shape()[0]])
        [X_ARR, Xbatch, i] = tf.while_loop(condition, body, [X_ARR, Xbatch, i],
                                           shape_invariants=[tf.TensorShape([None, None]),
                                                             Xbatch.get_shape(), i.get_shape()])

        x = tf.reduce_sum(X_ARR, 0)

        output_data.append(tf.expand_dims(x, 0))

    output_data = tf.stack(output_data, 0)

    return output_data

def rickerWavelet(scale, sampleCount):
    scale = tf.to_float(scale)
    sampleCount = tf.to_float(sampleCount)

    def rickerWaveletEquationPart(time):
        time = tf.to_float(time)

        tSquare = time ** 2.
        sigma = 1.
        sSquare = sigma ** 2.

        # _1 = 2 / ((3 * a) ** .5 * np.pi ** .25)
        _1a = (3. * sigma) ** .5
        _1b = np.pi ** .25
        _1 = 2. / (_1a * _1b)

        # _2 = 1 - t**2 / a**2
        _2 = 1. - tSquare / sSquare

        # _3 = np.exp(-(t**2) / (2 * a ** 2))
        _3a = -1. * tSquare
        _3b = 2. * sSquare
        _3 = tf.exp(_3a / _3b)

        return _1 * _2 * _3

    unscaledTimes = tf.to_float(tf.range(tf.to_int32(sampleCount))) - (sampleCount - 1.) / 2.
    times = unscaledTimes / scale
    unscaledSamples = rickerWaveletEquationPart(times)
    samples = unscaledSamples * scale ** -.5

    return samples

def conv1DWavelet(wav, waveletWidth, waveletEquation):
    kernelSamples = waveletWidth * 10
    kernel = waveletEquation(waveletWidth, kernelSamples)
    kernel = tf.reshape(kernel, tf.pack([kernelSamples, 1, 1, 1]))

    conv = tf.nn.conv2d(wav, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.squeeze(tf.squeeze(conv))

    return conv

def matrixnormalise(matrix, a, b):
    newmatrix = np.empty(shape=matrix.shape)
    Xmin = np.min(matrix, axis=0)
    Xmax = np.max(matrix, axis=0)

    for i in range(matrix.shape[0]):
            newmatrix[i] = a + ((matrix[i]-Xmin)*(b-a))/(Xmax-Xmin)

    return newmatrix

def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))

    return (100. * num_correct/batch_predictions.shape[0])

def SGSmoothing(EEG):
    return sgl.savgol_filter(EEG,11,2)

def preprocess(dataframe):

    new_df = pd.DataFrame(columns=dataframe.columns)

    for i in range(0, dataframe.shape[0]):
        row =[]
        for j in dataframe.columns:

            EEG = dataframe[j][i]
            EEG = SGSmoothing(EEG)

            row.append(EEG)

        new_df.loc[i] = row

    return new_df