import matplotlib.pyplot as plt
import tqdm
from hyperparameters import *
from parameters import *
from tools import *
from waveletmodel import Wavenet

sess = tf.Session()

X_train, Y_train, X_test, Y_test = loadDataFrames(TS_length)
target_size = int(max(Y_train) + 1)

model_output = Wavenet(X_input)
test_model_output = Wavenet(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(model_output), tf.squeeze(Y_target)))

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
train_acc = []
test_acc = []

prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

for i in tqdm.tqdm(range(generation)):

    rand_index = np.random.choice(len(X_train), size=batch_size)
    rand_x = X_train[rand_index]
    rand_y = Y_train[rand_index]

    traindict = {X_input: rand_x, Y_target: rand_y}
    sess.run(train_step, feed_dict=traindict)

    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=traindict)
    temp_train_accuracy = get_accuracy(temp_train_preds, rand_y)/100
    train_loss.append(temp_train_loss)
    train_acc.append(temp_train_accuracy)

    if (i+1) % eval_every ==0:

        eval_index = np.random.choice(len(X_test), size=evaluation_size)
        eval_x = X_test[eval_index]
        eval_y = Y_test[eval_index]

        test_dict = {eval_input: eval_x, eval_target: eval_y}
        test_preds = sess.run(test_prediction, feed_dict=test_dict)
        temp_test_acc = get_accuracy(test_preds, eval_y)/100
        test_acc.append(temp_test_acc)


plt.plot(train_acc, label='Accuracy', color='blue')
plt.plot(train_loss, label='Loss', color='red')
plt.plot(test_acc, label='Test', color='green')
plt.legend()
plt.show()

