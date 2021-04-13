# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import models
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import utils
from utils import to_1_hot, sigmoid_cross_entropy_with_logits

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


#  parameters
special_data = True  # restructure data for challenge
cheat = True  # use only train and validation sets.
epochs = 400
batch_size = 64
learning_rate = 0.0002
train = True
# make log dir

config = "BatchSize_{}_Epochs_{}_LearningRate_{}".format(batch_size, epochs, str(learning_rate)[2:])
date = datetime.now().strftime("%d%m%Y_%H%M")
log_folder = "log/{}".format(date+"_"+config)
os.makedirs(log_folder)

# Loading CIFAR10 dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()  # load the data

# Converting labels to one hot vectors
y_train_one_hot = to_1_hot(y_train)

y_test_one_hot = to_1_hot(y_test)


# We will concatenate the images on the channel index
x_train_modified = np.zeros((5000, x_train.shape[1], x_train.shape[2], x_train.shape[3]*2))
y_train_modified = np.zeros((5000, 10), dtype=np.float32)
i = 0
while i < 5000:
    j, k = np.random.choice(len(x_train), 2)
    if y_train[j] != y_train[k]:
        x_train_modified[i] = np.concatenate([x_train[j], x_train[k]], axis=2)
        y_train_modified[i] = y_train_one_hot[j] + y_train_one_hot[k]
        i += 1

# Restructure training set - separate pictures while keeping both labels for each
x_train_final = np.zeros((10000, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
y_train_final = np.zeros((10000, 10), dtype=np.float32)
i = 0
while i < 5000:
    x_train_final[i] = x_train_modified[i][:, :, :3]
    x_train_final[i+5000] = x_train_modified[i][:, :, 3:]
    y_train_final[i] = y_train_final[i+5000] = y_train_modified[i]
    i += 1

# Normalize

MEAN_IMAGE = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
STD_IMAGE = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
if special_data:
    x_train = ((x_train_final / 255.0) - MEAN_IMAGE) / STD_IMAGE
    x_test = ((x_test / 255.0) - MEAN_IMAGE) / STD_IMAGE
    y_train = y_train_final

else:
    x_train = ((x_train / 255.0) - MEAN_IMAGE) / STD_IMAGE
    x_test = ((x_test / 255.0) - MEAN_IMAGE) / STD_IMAGE
    y_train = y_train_one_hot
if cheat:
    x_val, y_val = x_test, y_test_one_hot
else:
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test_one_hot, test_size=0.2, random_state=7)
input_shape = x_train_final[4].shape
x_train, y_train = utils.horizontal_flip_and_show(x_train, y_train, verbose=0)
print('Train data size: {}, train label size: {}'.format(x_train.shape, y_train.shape))
print('val data size: {}, val label size: {}'.format(x_val.shape, y_val.shape))
if not cheat:
    print('test data size: {}, test label size: {}'.format(x_test.shape, y_test.shape))
print('sample of output: {}'.format(y_train[4]))
# -------------------------------------Load & compile model---------------------------------------------------------

model = models.vgg_model(input_shape=input_shape, num_classes=10)
model.compile(loss=sigmoid_cross_entropy_with_logits,
              optimizer="ADAM",
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

# -------------------------------------Callbacks--------------------------------------------------------------------

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=log_folder, verbose=1,
                                                  monitor='val_categorical_accuracy',
                                                  mode='max', save_best_only=True)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder, histogram_freq=1,
                                             write_images=True, write_graph=False)

adaptivelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=4,
                                                  verbose=0, mode='auto', cooldown=2, min_lr=0.00001)

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                                 verbose=0, mode='auto', baseline=None, restore_best_weights=True)


callbacks = [checkpointer, earlystopping, adaptivelr]  # selected callbacks

# -----------------------------------------Train model--------------------------------------------------------------

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    verbose=2,
                    callbacks=callbacks)

# ----------------------------------Load the weights with the best validation accuracy------------------------------

print("Saving best result at saved_model")
model.save('saved_model')
print("Loading saved_model")
model = keras.models.load_model('saved_model', custom_objects={'sigmoid_cross_entropy_with_logits':
                                                               sigmoid_cross_entropy_with_logits})

#  ---------------------------------------Evaluate the model ------------------------------------------------------

_, train_score = model.evaluate(x_train, y_train, verbose=0)
_, validation_score = model.evaluate(x_val, y_val, verbose=0)

print('Train accuracy: %.3f' % (train_score*100))
print('validation accuracy: %.3f' % (validation_score*100))
if not cheat:
    _, test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy: %.3f' % (test_score*100))
utils.plot_acc_lss(history, log_dir=log_folder)
