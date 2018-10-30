# Train Deep Convolutional Network on MNIST using MONITORED TRAINING SESSION in SYNCHRONOUS DISTRIBUTED WAY

In MNIST-03 we have developed distributed training in asynchronous way. This is the fastest distributed way but could present several issues like gradient stale. In order to avoid this problem TensorFlow allows user to use synchronous distributed training. In this mode each worker has to wait for the other when a gradient calculation step finishes. Then the chief worker update the gradients using calculated gradients of each worker and nex iteration step can take place.

For this example the original DeepMNIST_DistributedTRAIN.py form MNIST-03 TUTORIAL was modified and moved to:

	DeepMNIST_DistributedTRAIN_Sync.py

DeepMNIST_DistributedTRAIN_Sync.py

*************************************************************************************************

Basic change in this script are:

	Lines 51-58: Here we define the optimizer to use (line 52) and pass it to the tf.train.SyncReplicasOptimizer.
	This function needs:
		* Optimizer.
		* replicas_to_aggregate=number of replicas to aggregate for each variable update.
		In our case will be equal to the number of workers.
		* total_num_replicas: Total number of tasks/workers/replicas, could be different from replicas_to_aggregate.
			+ If total_num_replicas > replicas_to_aggregate: it is backup_replicas + replicas_to_aggregate. 
			+ If total_num_replicas < replicas_to_aggregate: Replicas compute multiple batches per update to variables
			+ In our case will be equal to number of workers.
	Lines 59-62: Here we define the training operation: It will call the minimize method of the synchronous optimizer
	This minimize operation need following inputs:
		* Loss function to minimize
		* global_step
		* aggregation_method in order to compute the final gradient using the gradients calculated by the replicas.
	Line 76: Here we divide the batch_size by the number of workers.
	Each worker proccess its correspondent part of the batch_size. This could be changed.
	Line 80-81: Here we create a Hook from the synchronous optimizer that allow TF syncronyze workers 
	inside the training loop. This hook need the inputs:
		* is_chief: True/False. Synchronizing hook need to know what worker is chief for dealing with synchro barriers.
		* num_tokens=0. We need this in order to create a synchro barrier at the begginning of the training.
		If we do not give num_tokens when enter in training loop, training begins inmediately 
		without test if all workers are ready.
		If, for example, chief worker is ready and the other are not, the chief begins to calculate gradients.
		When chief worker calculates the necesary number of gradients it will update the total gradient.
		So until all workers are online the training will be done only by the chief.
		This could be not optimal if user divides the dataset and each worker trains a different part of dataset
		(in this example all worker trains over the same dataset).
	Line 97: here the hook for final summary will be done only by the chief worker 
	because all workers have the same weights and bias of the model 
	(in asynchronous way workers will have different versions of weights and bias).


LaunchTFServerWithHooks.py

*************************************************************************************************
In order to keep the functionality of closing Distributed TF server gracefully in synchronous training is necesary make a little change in  LaunchTFServerWithHooks.py:

	Line 60: time.sleep(180): here we add a waiting time of 3 minutes since PS dequeue 
	the TF queues from all the workers. With this waiting before close the PS session 
	we allow chief worker to finished all the queues that synchronize operation requires.
