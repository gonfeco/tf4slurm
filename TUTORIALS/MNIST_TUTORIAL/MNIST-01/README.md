# Original TensorFlow for MNIST with Deep Networks running

Here we begin the adaptation of the TF MNIST deep code for running in parallel.
Steps:

	-Code was downloaded from (TensorFlow v1.3):
	https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/mnist/mnist_deep.py 
	-File is renamed to: mnist_deep.py file.  
	-In order to get more functionalities we created: mnist_deep_modification.py 
	-Basic Modifications of mnist_deep_modification:
		* User can provided the folder to store MNIST data (FLAG: --data_dir).
		* User can provided the folder to store Trained Model (FLAG: --model_dir).
		* User can provided batch size (FLAG: -batch_size).
		* User can provided maximum number of iterations (FLAG: -Iterations).
	-Bash script (BASH_MNIST_DEEP_Original.sh) to submit training is provided.
	User can easily make changes in bash script in order to modify before FLAGS.

