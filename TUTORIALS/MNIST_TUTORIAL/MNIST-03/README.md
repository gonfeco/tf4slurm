# Train Deep Convolutional Network on MNIST using MONITORED TRAINING SESSION in DISTRIBUTED WAY

Here we develop distributed training based on the scripts from MNIST-02. In order to keep the code as clean as possible we have created following scripts:

	- LaunchTFServerWithHooks.py: This script deals with the configuration and execution of the Distributed TF Server.
	- DeepMNIST_DistributedTRAIN.py: This script was created from the training part of the previous code 
			(MNIST-02/DeepMNIST_MonitoredSession.py). 


LaunchTFServerWithHooks.py

#############################################################################################

This scripts uses ServerDictionary and DistributedTFQueueHook modules from the tf_for_slurm package for creating the Distributed TensorFlow server (they are imported in lines 25 and 26). Distributed TF server is mandatory for deploy a distributed Training. User can provided to the script (using argument parser):

	--data_dir: folder to store MNIST data 
	--model_dir:folder to store Trained Model 
	-batch_size:batch size
	-Iterations:maximum number of iterations.
		-ps: number of parameter servers used in the training
		-workers: number of workers used for training.
		--NoIB: do not use Infini Band.

Important parts:

	- Lines 43-47: Here the List of task (ListOfTFTasks) for the Distributed TF server is created. 
		* It is MANDATORY that: size(ListOfTFTasks)=Number of Tasks.
		* Each element is a string that could be "ps" or "worker". 
		* In this code all the elements will be "worker" except the first PS elements that will be "ps". 
		* This list is important because determines what machines will be "ps" and what will be "workers".
		* ABOUT THE ListOfTFTasks:
			1.-This list is very important when user wants to use several nodes with several tasks by node. 
			2.-The three posible cases for launching a distributed training are:
				2.1- ONE NODE, N TASKS. Here the order of the elements of the list is not important.
				2.2- N NODES, 1 TASK/PERNODE. This case is identical to the previous one. 
				2.3- N NODES, M TASKS/PERNODE: Number  of total tasks=N*M. In this case the M first elements of the list will be related with first node.
			Next M wiht the second one node and so on.
			So depending on the element the user puts the "ps" the "ps" goes to one node or to other.
			If user wants more than one "ps" user can distributed the "ps"s between the different nodes 
			by carefully asigning "ps" in the ListOfTFTasks.
			EXAMPLE: Number of Nodes=2. Task/pernode=4. Number of Total Tasks: 2*4=8. So If user wants PS=2:
				EXAMPLE-1: ['ps','ps', 'worker', 'worker', 'worker', 'worker', 'worker', 'worker'] -> The PSs will be in the first Node!!
				EXAMPLE-2: ['ps', 'worker', 'worker', 'worker','ps', 'worker', 'worker', 'worker'] -> One PS in first node other ps in second node.

	- Line 48: This is the call to the GetServerDictionary() function from the ServerDictionary module.Here each task gets:
		* ServerDictionary: python dictionary to create the Distributed TF server dictionary.
		* (job_type,task_index): identification of the machine in the ServerDictionary:
			+ job_type: "ps" or "worker".
			+ task_index: index of local machine in the list of "ps" or "worker" (depending on job_type) in the ServerDictionary.
	- Lines 49-55: Creates the cluster and the Server for running TF in distributed way.
	Finally each local machine could be a "ps" or a "worker" (this is given by job_type):
	- ps (lines 56-63): if the machine is a "ps" then a parameter server is created. 
	This server will run until all workers finished their job. This behaviour is not the original in Distributed TF. 
	Its an implementation of Yaroslab Butalov for running parameter servers using TF Queues.
		* Line 58: The TF queues are created by invoking create_done_queue() from DistributedTFQueueHook module in each PS task.
		* Lines 60-62: ALl the PS tasks try to dequeue their correspondent TF queue. 
			This block the code excution in the PS until all PS tasks dequeue their correspondent TF qeueues.     
	- worker (lines 64-66): If the machine is a worker: 
		* First we create a TF Hook using QueueManagementHook class from DistributedTFQueueHook module. 
		* In the begin part of this class an enqueue operation on the worker over all the TF queues created by the PS tasks is defined. 
		* In the end part this enqueue operation is executed. 
		* In line 66 the function DistributedTrain() from the DeepMNIST_DistributedTRAIN.py is invoked. 
		* User should be aware that the TF Hook from the QueueManagementHook is passes to this function.

It is important to highlihgt here that this script can be used for deploy any training of any distributed model.The LaunchTFServerWithHooks.py script only launchs the Distributed TF server .Only  in the final line the the training function is called. So if user implements its own Distributed Training function only needs call it in the final line of the script. (REMEMBER TO INCLUDE THE import!!!!). Of course user can add more positional arguments or even delete the 4 first arguments(script only needs -ps and -worker positional arguments) in order to adapt their training scripts.

DeepMNIST_DistributedTRAIN.py.

#############################################################################################
This script implements the Distributed Training. In fact with the MNIST-02/DeepMNIST_MonitoredSession.py script we have all that we need the Distributed Training and only some minor modifications are needed. The main function is DistributedTrain (that is called by the last line of LaunchTFServerWithHooks.py). Important change with respect to the original file:
		
	- Line 37: here we stablish what worker task will be the chief (is_chief=(task_index == 0))
	- Line 38: master=TFServer.target this is needed for the monitored session.
	- Line 39: This a tf.device block that uses tf.train.replica_device_setter to distibuted the model.
	Variables will be assigned to PS and operations to workers in an automatic way.
	Under this block we put lines 47 to 71 of the MNIST-02/DeepMNIST_MonitoredSession.py script
	Basically under this "with" block user should put:
		* Model definition (line 47).
		* Loss definition (lines 48-53)
		* Optimizer defintion and training operation (lines 55-57)
		* Metrics definition (lines 59-63)
		
	- Line 73: Here we create the list of Hooks and put the QueueHook (that is an input of the DistributedTrain function).
	With this Hook we can use the Yaroslav Butalov solution to close the Distributed TF server gracefully. 
		
	- Line 75: Here we create a hook for stoping training using tf.train.StopAtStepHook(last_step=FLAGS.Iterations).
	IMPORTANT: Read README_SynchronizationProblems.txt
	
	- Lines 76-87: Here we take the Hooks for validation during training and put them inside an "if" block statement.
	We only want that chief worker make this training validations (the other workers only do optimizer operations).
		
	- Line 90: Commented call to BarrierOnChiefHook class for synchronization workers at begining of the training: 
		SEE: README_SynchronizationProblems.txt
		
	- Lines: 92-100: Here is the call to tf.train.MonitoredTrainingSession. And the only change is that we need to give the master (line 38) as input. The others inputs remain the sames.

AND THAT'S ALL!!! NOW YOU CAN GO TO DISTRIBUTED TF!!!!

BASH_Distributed_MNIST.sh

#############################################################################################

Bash script (BASH_Distributed_MNIST.sh) to submit training is provided.
User can easily make changes in bash script in order to modify before FLAGS.
ABOUT THE LAUNCHING SCRIPT.	
User should be aware about the RAM memory that is going to be assigned to each task. The script makes memory calculation depending on the FT2 partition where the job is submitted. The script calculates automatically RAM for each task: MEMPERTASK (line 43). The problem is that TF needs lot of RAM so if user wants only one core per task (the default case of the script) the dedicated RAM could be not enough and the trains could fail. Hence we set the memory per task to 10G (line 50). If user want to use more cores per tasks and use the automatic RAM reservation should uncomment line 49 and comment line 50. 


Hooks.py

#############################################################################################

Here we took the Hooks.py scritp from MNIST-02 and added following Hooks to increase functionalities:

	- BarrierOnChiefHook: this Hook could be used to synchronization of workers before training begins. 
	(See README_SynchronizationProblems.txt)
	- PrintStepHook: This Hook could be used to print in each worker the actual global step. 
	This Hook is not called from DeepMNIST_DistributedTRAIN.py.
