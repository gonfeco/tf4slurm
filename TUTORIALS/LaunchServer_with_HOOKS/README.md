# LaunchServer_with_HOOKS

In this example a Distributed TensorFlow server is created and executed using the Yaroslav Bulatov solution to bringing down the parameter server. This example use the 2 modules of the tf_for_slurm package: ServerDictionary and DistributedTFQueueHook
This example has three scripts:

	LaunchTFServer_Hook.py: this python script creates and executes a Distributed TF Server.
	SleepExample.py: python script that create graph and run a TF session.
	BASH_LaunchTFServer_WithHOOK.sh: bash script used for submitting pyhton script to FT2.

**************************************************************************************************************************

LaunchTFServer_Hook.py

This script creates a dsitributed TF server using ServerDictionary module. Important Lines:

	-line 8: import ServerDictionary from tf_for_slurm package.
	-line 9: import DistributedTFQueueHook from tf_for_slurm package.
	-lines 26-30: Create ListOfTFTasks for the Distributed TF server is created:It is MANDATORY that: size(ListOfTFTasks)=Number of Tasks:
		* Each element is a string that could be "ps" or "worker". 
		* In this code all the elements will be "worker" except the first PS elements that will be "ps". 
		* This list is important because determines what machines will be "ps" and what will be "workers".
		* ABOUT THE ListOfTFTasks:
			1.-This list is very important when user wants to use several nodes with several tasks by node. 
			2.-The three posible cases for launching a distributed training are:
				2.1- ONE NODE, N TASKS. Here the order of the elements of the list is not important.
				2.2- N NODES, 1 TASK/PERNODE. This case is identical to the previous one. 
				2.3- N NODES, M TASKS/PERNODE: Number  of total tasks=N*M. In this case the M first elements
				of the list will be related with first node.Next M wiht the second one node and so on. 
				So depending on the element the user puts the "ps" the "ps" goes to one node or to other.
				If user wants more than one "ps" user can distributed the "ps"s between the different nodes
				by carefully asigning "ps" in the ListOfTFTasks.
				EXAMPLE: Number of Nodes=2. Task/pernode=4. Number of Total Tasks: 2*4=8. So If user wants PS=2:
				EXAMPLE-1: ['ps','ps', 'worker', 'worker', 'worker', 'worker', 'worker', 'worker'] -> The PSs will be in the first Node!!
				EXAMPLE-2: ['ps', 'worker', 'worker', 'worker','ps', 'worker', 'worker', 'worker'] 
								-> One PS in first node other ps in second node.
	-line 31: This is the call to the GetServerDictionary() function from the ServerDictionary module.Here each task gets:
		* ServerDictionary: python dictionary to create the Distributed TF server dictionary.
		* (task_type,task_index): identification of the machine in the ServerDictionary:
			+ task_type: "ps" or "worker".
			+ task_index: index of local machine in the list of "ps" or "worker" (depending on job_type) in the ServerDictionary.
	-lines 42-40: Creates the cluster and the Server for running TF in distributed way.
	
Finally each local machine could be a "ps" or a "worker" (this is given by task_type):
	
	- ps (lines 41-47): if the machine is a "ps" then a parameter server is created. This server will run until all workers finished their job. This behaviour is not the original in Distributed TF. Its an implementation of Yaroslab Butalov for running parameter servers using TF Queues.
		* Line 42: The TF queues are created by invoking create_done_queue() from DistributedTFQueueHook module in each PS task.
		* Lines 44-46: ALl the PS tasks try to dequeue their correspondent TF queue. This block the code excution in the PS until all PS tasks dequeue their correspondent TF qeueues.
	- worker (lines 47-49): If the machine is a worker: 
		* Line 48: a TF Hook using QueueManagementHook class from DistributedTFQueueHook module is created.
			+In the begin part of this class an enqueue operation on the worker over all the TF queues created by the PS tasks is defined. 
			+In the end part this enqueue operation is executed.
		* Line 49: Here the SleepExample function from SleepEcample.py is invoked. Users can change this line and call to their custom TF code but should be aware that is mandatory to pass folowing information:
			+ task_index
			+ cluster: definition of the Distributed TF cluster
			+ server: Distributed TF server
			+ QueueHook: TF Hook to close gracefully Distributed TF server when all workers finish their jobs.

**************************************************************************************************************************
SleepExample.py


	Line 50: here we stablish what worker task will be the chief (is_chief=(task_index == 0))
	Line 51: master=server.target this is needed for the monitored session.
	Line 52-61:This a tf.device block that uses tf.train.replica_device_setter to distibuted the model.
	line 59: global_step is created in order to all workers know the step iteration. 
	line 61: a increment operation over the global step is defined.
	Line 63: Here we create a hook for stoping training at 10 iterations using tf.train.StopAtStepHook()
	Lines: 65-79: Here is the call to tf.train.MonitoredTrainingSession. Each iteration wait ten seconds and executes increase operation over global_step. 

**************************************************************************************************************************
BASH_LaunchTFServer_WithHOOK

This script submits the pyhton script to FT2. User can configure what modules used for the job by giving arguments to the bash script. Following posible arguments are shown: 

	sbatch BASH_LaunchTFServer_WithHOOK.sh -> use TensorFlow v1.7.0 and Python 2.7.14
	sbatch BASH_LaunchTFServer_WithHOOK.sh 10 -> use TensorFlow v1.0.0
	sbatch BASH_LaunchTFServer_WithHOOK.sh 12 -> use TensorFlow v1.2.0
	sbatch BASH_LaunchTFServer_WithHOOK.sh 13 -> use TensorFlow v1.3.0
	sbatch BASH_LaunchTFServer_WithHOOK.sh 17 -> use TensorFlow v1.7.0 and Python 2.7.14
	sbatch BASH_LaunchTFServer_WithHOOK.sh 17 2 -> use TensorFlow v1.7.0 and Python 2.7.14
	sbatch BASH_LaunchTFServer_WithHOOK.sh 17 3 -> use TensorFlow v1.7.0 and Python 3.6.5




