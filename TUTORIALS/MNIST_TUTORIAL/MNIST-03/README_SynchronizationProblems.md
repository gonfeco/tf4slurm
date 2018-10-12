# PROBLEMS WITH WORKER SYNCHRONIZATION

In MNIST-03 an Asynchronous Distributed training was implemented.Some problems raise about the way that the training finished that user should be aware.

	In line 71 of DeepMNIST_DistributedTRAIN.py we use the StopAtStepHook for stopping the training loop.
	User can pass to the hook two different inputs (that are mutually exclusive):
		- num_steps
		- last_step	
	TensorFlow by default uses num_steps but our script uses last_step instead. 
	
A bad behaviour when uses num_steps in the StopAtStepHook was detected:
Usually one of the workers (in general the chief worker) is online earlier than the others. In asynchronous training this fastest worker begins to train without waiting for the other workers.
		The firsts steps of training will be done only by this active worker.
		While other workers were online their continue the training from the global_step they see.
			EXAMPLE: slower worker can be online at global_step=200 and it continue training from this step.
		If num_steps is used in StopAtStepHook: following behaviour raises:
			-Faster worker: when global_step is higher than num_steps it stop its training.
			-Other workers: when global_step is higher than num_steps they continue the training.
				EXAMPLE: If slower worker was online in global_step=200 it finished the training at num_steps+200.
	There are two ways of avoid this behaviour:
		1.- Use last_step in StopAtStepHook:
			If global_step is higher than last_step then all the workers finished their trainings. 
			This method works but could be a problem if user wants that all the workers made the same training iterations
			(for example if each worker deals with one part of the dataset).
		2.- Create a synchronization Barrier at the beggining of the training that waits until all the workers are online.
			In this case all the workers begins the training at the same time and both inputs of StopAtStepHook can be used.
	In order to implements the last solution a new Hook class was implemented in the Hooks.py script:
		- BarrierOnChiefHook(tf.train.SessionRunHook).
			This Hooks works in a similar way than the QueueManagementHook class of DistributedTFQueueHook.py script.
			In the BarrierOnChiefHook class the dequeue operation its implemented in the chief worker.
			The other workers implements a enqueue operation. 
			So untill all workers were online the chief worker can not do the dequeue operation and will be blocked.
			This Hook has to be called before the tf.train.MonitoredTrainingSession block and needs the following inputs:
				* is_chief= True/False value dependig if the worker is chief or not.
				* chief_task_index= Integer. Task index of the chief_worker (usually is 0)
				* NumberOfWorkers= Integer. Number of total workers.
			In line 86 of DeepMNIST_DistributedTRAIN.py we include a commented call to this Hook.
			If user want to use only has to uncomment this line. 
	If user want to see this bad behaviour can use the PrintStepHook from Hooks.py and added to the DeepMNIST_DistributedTRAIN.py
	as a typical Hook. This hooks prints for each worker the correspondent global_step in each iteration.

