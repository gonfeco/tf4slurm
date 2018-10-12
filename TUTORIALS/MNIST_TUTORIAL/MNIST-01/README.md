# Original TensorFlow for MNIST with Deep Networks running

Here we begin the adaptation of the TF MNIST deep code for running in parallel.
Steps:
1.-Code was downloaded from (TensorFlow v1.3):
	https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py 
2.-File is renamed to: mnist_deep.py file.  
3.-In order to get more functionalities we created: mnist_deep_modification.py 
4.-Basic Modifications of mnist_deep_modification:
	4.1-User can provided the folder to store MNIST data (FLAG: --data_dir).
	4.2-User can provided the folder to store Trained Model (FLAG: --model_dir).
	4.3-User can provided batch size (FLAG: -batch_size).
	4.4-User can provided maximum number of iterations (FLAG: -Iterations).
5.-Bash script (BASH_MNIST_DEEP_Original.sh) to submit training is provided.
	User can easily make changes in bash script in order to modify before FLAGS.

