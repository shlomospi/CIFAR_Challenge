import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import os


def to_1_hot(labels):
    """
    imubit implementation of 1 hot encoding
    Args:
        labels: label tensor as int

    Returns: label tensor as bool

    """
    return np.eye(10)[labels][:, 0, :]


def plot_acc_lss(history_log, log_dir=None, verbose=1):
    """
    plot loss and acc via plt
    Args:
        history_log: history file of training session
        log_dir: dir to save the plot
        verbose: 1 for showing the plot 0 for not showing the plot

    Returns: NaN

    """

    # ----summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_log.history['categorical_accuracy'])
    plt.plot(history_log.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(b=True, which='Both', axis='both')
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history_log.history['loss'])
    plt.plot(history_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(b=True, which='Both', axis='both')

    if log_dir is None:
        while True:
            given_dir = input("To save training plot, provide dir location. leave empty to skip")
            if given_dir == "":
                break
            else:
                try:
                    plt.savefig('{}/training_plots.png'.format(given_dir))
                except ValueError:
                    print("could not save in provided dir: {}".format(given_dir))
                    continue
                else:
                    break
    else:

        plt.savefig('{}/training_plots.png'.format(log_dir))

    if verbose >= 1:
        plt.show()


def sigmoid_cross_entropy_with_logits(target, output):
    """
    wrapper-workaround for loading model. find cleaner solution
    Args:
        target: ground truth
        output: graph output

    Returns: sigmoid cross entropy loss from logits, TF implementation

    """

    return tf.nn.sigmoid_cross_entropy_with_logits(target, output)


def horizontal_flip_and_show(data, labels, verbose=0):
    """
    tf.image.flip_left_right wrapper. adds left right augmented images and labels
    :param data: image tensor
    :param labels: one hot label tensor
    :param verbose: 1 to show random example
    :return: data with flipped addition, new labels
    """
    with tf.device('/device:cpu:0'):
        fliped_data = tf.image.flip_left_right(data)
        if verbose >= 1:

            rand = random.randint(0, 1000)
            plt.subplot(1, 2, 1)
            plt.title('Original Image #{}'.format(rand))
            plt.imshow(data[rand, :, :, :])
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title('Augmented Image #{}'.format(rand))
            plt.imshow(fliped_data[rand, :, :, :])
            plt.axis('off')
            plt.show()

    return tf.concat([data, fliped_data], 0), tf.concat([labels, labels], 0)


def check_folder(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir