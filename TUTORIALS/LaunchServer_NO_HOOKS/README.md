# LaunchTFServer_NoHOOKS

In this example a Distributed TensorFlow (TF) server is created and executed in the typical TF way. This example use the ServerDictionary module from the tf_for_slurm package. This example has two scripts:

	LaunchTFServer_NoHOOKS.py: this python script creates and executes a Distributed TF Server.
	BASH_LaunchTFServer_NoHOOKS.sh: bash script used for submitting pyhton script to FT2.
	
**************************************************************************************************************************

LaunchTFServer_NoHOOKS

This script creates a dsitributed TF server using ServerDictionary module. Important Lines:

	-line 8: import ServerDictionary from tf_for_slurm package.
	-lines 25-29: Create ListOfTFTasks for the Distributed TF server is created:
		*It is MANDATORY that: size(ListOfTFTasks)=Number of Tasks.
		*Each element is a string that could be "ps" or "worker".
		*In this code all the elements will be "worker" except the first PS elements that will be "ps". 
		*This list is important because determines what machines will be "ps" and what will be "workers".
		*ABOUT THE ListOfTFTasks:
			1.-This list is very important when user wants to use several nodes with several tasks by node. 
			2.-The three posible cases for launching a distributed training are:
				2.1- ONE NODE, N TASKS. Here the order of the elements of the list is not important.
				2.2- N NODES, 1 TASK/PERNODE. This case is identical to the previous one. 
				2.3- N NODES, M TASKS/PERNODE: Number  of total tasks=N*M. In this case the M first elements of the list will be related with first node.Next M wiht the second one node and so on. So depending on the element the user puts the "ps" the "ps" goes to one node or to other. If user wants more than one "ps" user can distributed the "ps"s between the different nodes by carefully asigning "ps" in the ListOfTFTasks.
				EXAMPLE: Number of Nodes=2. Task/pernode=4. Number of Total Tasks: 2*4=8. So If user wants PS=2:
				EXAMPLE-1: ['ps','ps', 'worker', 'worker', 'worker', 'worker', 'worker', 'worker'] -> The PSs will be in the first Node!!
				EXAMPLE-2: ['ps', 'worker', 'worker', 'worker','ps', 'worker', 'worker', 'worker'] 
								-> One PS in first node other ps in second node.
	-lines 30-33: This is the call to the GetServerDictionary() function from the ServerDictionary module.Here each task gets:
		*ServerDictionary: python dictionary to create the Distributed TF server dictionary.
		*(task_type,task_index): identification of the machine in the ServerDictionary:
			+task_type: "ps" or "worker".
			+task_index: index of local machine in the list of "ps" or "worker" (depending on job_type) in the ServerDictionary.
	-lines 34-41: Creates the cluster and the Server for running TF in distributed way.
Finally each local machine could be a "ps" or a "worker" (this is given by task_type):
	
	ps (lines 44-45): if the machine is a "ps" then a parameter server is created. In this case the join method of the server is invoked. This method keeps the Distributed server TF running  forever until user or queue system kills it
	
**************************************************************************************************************************

BASH_LaunchTFServer_NoHOOKs

This script submits the pyhton script to CESGA Finis Terrae II Slurm system. User can configure what modules used for the job by giving arguments to the bash script. Following posible arguments are shown:

	sbatch BASH_LaunchTFServer_NoHOOKS.sh -> use TensorFlow v1.7.0 and Python 2.7.14
	sbatch BASH_LaunchTFServer_NoHOOKS.sh 10 -> use TensorFlow v1.0.0
	sbatch BASH_LaunchTFServer_NoHOOKS.sh 12 -> use TensorFlow v1.2.0
	sbatch BASH_LaunchTFServer_NoHOOKS.sh 13 -> use TensorFlow v1.3.0
	sbatch BASH_LaunchTFServer_NoHOOKS.sh 17 -> use TensorFlow v1.7.0 and Python 2.7.14
	sbatch BASH_LaunchTFServer_NoHOOKS.sh 17 2 -> use TensorFlow v1.7.0 and Python 2.7.14
	sbatch BASH_LaunchTFServer_NoHOOKS.sh 17 3 -> use TensorFlow v1.7.0 and Python 3.6.5
**************************************************************************************************************************
