# Running MNIST with Deep Networks using MONITORED TRAINING SESSION

When using Distributed TF the recomended way is training with tf.train.MonitoredTrainingSession.
Here we adapted MNIST-01 python scripts using this MonitoredTrainingSession for running in Serie.
Modifications of the previous code (MNIST-01/mnist_deep_modification.py):
In order to get the code as clear (and clean) as posiblle following scritps were created:
	
	-Model_For_mnist_deep.py: Here we coded the Deep Convolutional model.
	-DeepMNIST_MonitoredSession.py: Here we train the model of the previous script.

Model_For_mnist_deep.py

*************************************************************************************************

Here we copy the Deep Convolutional model from mnist_deep_modification.py. 
	
	lines 42-125 from  mnist_deep_modification.py moved to Model_For_mnist_deep.py

DeepMNIST_MonitoredSession.py

*************************************************************************************************
	
Here we train the model (that it is stored in the Model_For_mnist_deep.py script.
Important changes are:

	- Line 33: import Model_For_mnist_deep as Deep.
	- Line 50: loads model.
	- Line 52: global step is created using: tf.train.get_or_create_global_step().
	- Lines 59-61: Creates Optimizer, define training operation and global step is passed to this operation.
	- Line 101: Create a TensorFlow Hook to stop the training (StopAtStepHook) for FLAGS.Iterations training steps.
	- Lines 102-109: Training Loop using MonitoredTrainingSession:
	The global_step stores the step in the training loop and it will be mandatory in Distributed Training.

Minor changes and increasing functionality: We have removed from the original training loop the "if statement" used for monitoring training. Now we will monitor the training usign Customized TensorFlow Hooks (defined in Hooks.py).Instead of printing the monitorization in screen we will use the TensorBoard.These Hooks allow user to "hook" code into the TF session to several purpouses like metric evaluation. These Hooks are very useful in order to make validation when used distributed TF. Hooks allow code reusing.
Changes are:

	- Lines 69-72:Here we define the Summary operations we want to use to monitor the Training.
		This summary operations will be the CrossEntropy and the Accuracy.
	- Lines 76-85: Here we select a sub set of the Training Dataset for monitor purpouses.
		This is done to avoid long validation times. Monitor will be done always on the same data. 
		In original file step batch data was used for monitor (different data used for each monitor step). 
	- Lines 88-90: Here we call one of the customized hooks (NewSummarySaverHook) for monitor training. 
		We pass to the Hook the Summary operation (see 4.1) and the data used for summary. 
	- Lines 93-95: There is a commented Hook for performing monitoring on Testing Data.
	- Line 97: Here we call other customized Hook (FinalSummaryHook).
		This Hook print in screen results of Accuracy and Cross Entropy in the final training step usign the Testing Data.  
	- Line 92: Other similar Hook for usign de complete Training Dataset is commented. 
		User can uncomment before Hooks in order to get more monitoring functionalities. 

BASH_MNIST_DEEP_Monitored.sh.

*************************************************************************************************

This a bash script to submit training to the FT2 queue system.

