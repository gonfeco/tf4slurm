# tf4slurm
Integrating Distributed TensorFlow with Slurm queue systems
This Python Package was initially developed in Centro de Supercomputación de Galicia (CESGA, www.cesga.es) under the framework of the European Union’s H2020 research and innovation programme (grant agreement No 680481).
Complete description of this package and its successful application to an industrial case is provide in correspondent CESGA technical report: “Integrating Neural Network Parallel Training using Tensorflow with SLURM” (https://www.cesga.es/es/biblioteca/downloadAsset/id/803)
*****************************************************************************
Tree Folder Description:
There are 4 main folders in the package:

	1.-tf4slurm
	2.-LaunchServer_NO_HOOKS
	3.-LaunchServer_with_HOOKS
	4.-MNIST_TUTORIAL
	
************************************************************************************************

tf4slurm

The tf_for_slurm folder contains the Python Package mandatory for using distributed Training
capabilities of TensorFlow in a Slurm queue system. There are two different modules in the package: ServerDictionary and DistributedTFQueueHook.

ServerDictionary

All the basic functions to build the ServerDictionary for a Distributed TF Server are provided. When a Distributed TF training is required all the machines involved in the training need to load the "GetServerDictionary()" function of this script. This function needs following input:

	*TensorFlowServerTasks: List of strings: Lenght of list = number of machines. Each element should be "ps" o "worker".
  
  
The ouputs of the function are:

	DictionaryServer: python dictionary with the server configuration for the TF server: {"ps": [List of Addresses for "ps"],         "worker": [List of Addresses for "worker"]}. Same for all machines.
	task_type: string: "ps" or "worker". Depends of the machine.
	task_index: integer. Depends of the machine. Is the element of the machine in the list of the DictionaryServer depending of       the task_type.

DistributedTFQueueHook

This module defines the clase QueueManagementHook (based on tf.train.SessionRunHook) and the functions needed for closing gracefully the parameter server. This solution was based on the solution by Yaroslav Bulatov.
		
		https://gist.github.com/yaroslavvb/ea1b1bae0a75c4aae593df7eca72d9ca).
		
This solution creates a TF Queues on all the parameter servers ("ps") and the "ps" session try to dequeue them. Meanwhile the Queue is empty this operation blocks the "ps" server. In the "worker" the Hook of the script is created and when the training operation is finished the hook fills the "ps" queues. When "ps" queue are filled the dequeue operation can be finished and the "ps" can be closed gracefully.

2 bash scripts are included that are used by the submitting scripts (see LaunchServer_NO_HOOKS and LaunchServer_NO_HOOKS) to configure the modules into the FT2:

	*ModulesForRedHat6.7.sh
	*ModulesForRedHat7.5.sh

2 wraper scripts are included that are used for getting the IPs of the allocated nodes:

	*Wraper_NoIB.sh: Get the Ethernet IP addresses.
	*Wraper_IB.sh: Get Infini Band IP addresses.

These scritps must be called in the submision bash script before submit the TensorFlow code. The scripts get the IPs and write the them to an environment variable. This variable is readed in the ServerDictionary.py script for creating the TF Dictionary serer..In the LaunchServer_NO_HOOKS and LaunchServer_with_HOOKS the submitting bash scripts show how to use this scripts. 


**************************************************************************************************

LaunchServer_NO_HOOKS.

This folder contains all the scripts (python and bash scripts) to submit a typical Distributed TF server to FT2. Only uses the: ServerDictionary module from tf4slurm package. This example creates the TF server with "ps" and the "workers". The "ps" server will run forever until user o queue system kills the job. Folder contains a README with a detailed explanation of the scripts. Users can use the scripts of this folder for submitting their jobs by adding their code to the end of LaunchTFServer_NoHOOKS.py under the elif block.

LaunchServer_with_HOOK.

This folder contains all the scripts (python and bash scriprs) to submit the Distributed TF server with the Yaroslav Bulatov solution to FT2. Both package modules (ServerDictionary and DistributedTFQueueHook) are needed. The example creates the server and the when the workers finished their job the "ps" server is closed and the job will be finished. Folder contains a README with a detailed explanation of the scripts. Users can use the scripts of this folder for submitting their jobs by adding their code to the end of LaunchTFServer_with_HOOKS.py under the elif block. Users should remember to include the QueueHook in the call of their function.


MNIST_TUTORIAL

This Folder contains 4 different sub-folders with examples for deployment of a distributed Training of a Deep Convolutional Network using the MNIST dataset based on an example from TF webpage: mnist_deep.py file. Please follow the README.txt of the MNIST_TUTORIAL folder 


