## CIFAR10 with partial information about labels

This project is a challenge for an interview. Here, instead of using the original dataset, 5,000 pairs of images are selected for training, and new labels are created for each pair such that it is not known which label belongs to each picture.

# Approach

On the premise that inconsistent inaccuracies in the labels will not significantly hinder the ability of the model to generalize, it was decided to train one model with a simple VGG architecture. The data was restructured, and each picture was given both labels of the original pair. The final activation function was switched to sigmoidal to fit the new label structure. 

The model reached 77% accuracy on the validation set.

# How to run
Run
$ pip install requirements.txt
To load the trained model run:
$ python Main.py -p evaluate
To train a new model run:
$ python Main.py
It is possible to change the epoch number, batch size, learning rate, and log directory from the CLI. More documentation in the code.
