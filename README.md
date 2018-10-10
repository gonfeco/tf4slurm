# tf4slurm
Integrating Distributed TensorFlow with Slurm queue systems
This Python Package was initially developed in Centro de Supercomputación de Galicia (CESGA, www.cesga.es) under the framework of the European Union’s H2020 research and innovation programme (grant agreement No 680481).
Complete description of this package and its successful application to an industrial case is provide in correspondent CESGA technical report: “Integrating Neural Network Parallel Training using Tensorflow with SLURM” (https://www.cesga.es/es/biblioteca/downloadAsset/id/803)
*****************************************************************************
Tree Folder Description:
There are 2 main folders in the package: the code (tf4slurm) and the examples (TUTORIALS).
************************************************************************************************

tf4slurm

The tf_for_slurm folder contains the Python Package mandatory for using distributed Training
capabilities of TensorFlow in a Slurm queue system. There are two different modules in the package: ServerDictionary and DistributedTFQueueHook.

ServerDictionary

All the basic functions to build the ServerDictionary for a Distributed TF Server are provided. When a Distributed TF training is required all the machines involved in the training need to load the "GetServerDictionary()" function of this script. This function needs two inputs:

	*TensorFlowServerTasks: List of strings: Lenght of list = number of machines. Each element should be "ps" o "worker".
  
	*InfiniBand: True (if infiniband is used). False (communincations through ethernet). This is a custom option for CESGA Finis Terrae 2 system. You should check IP InfiniBand configuration of your system. 
  
The ouputs of the function are:

	DictionaryServer: python dictionary with the server configuration for the TF server: {"ps": [List of Addresses for "ps"],         "worker": [List of Addresses for "worker"]}. Same for all machines.
	task_type: string: "ps" or "worker". Depends of the machine.
	task_index: integer. Depends of the machine. Is the element of the machine in the list of the DictionaryServer depending of       the task_type.

DistributedTFQueueHook

This module defines the clase QueueManagementHook (based on tf.train.SessionRunHook) and the functions needed for closing gracefully the parameter server. This solution was based on the solution by Yaroslav Bulatov.
		
		https://gist.github.com/yaroslavvb/ea1b1bae0a75c4aae593df7eca72d9ca).
		
This solution creates a TF Queues on all the parameter servers ("ps") and the "ps" session try to dequeue them. Meanwhile the Queue is empty this operation blocks the "ps" server. In the "worker" the Hook of the script is created and when the training operation is finished the hook fills the "ps" queues. When "ps" queue are filled the dequeue operation can be finished and the "ps" can be closed gracefully.

**************************************************************************************************

TUTORIALS 

Several Tutorials that show how to use the tf_for_slurm package are provided in TUTORIALS folder. Under this path there are 3 subfolders: LaunchServer_NO_HOOKS, LaunchServer_with_HOOKS and MNIST_TUTORIAL.

LaunchServer_NO_HOOKS.

This example launches a typical Distributed TF server. Only uses the: ServerDictionary module from tf4slurm package. This example creates the TF server with "ps" and the "workers". The "ps" server will run forever until user o queue system kills the job. Folder contains the python script and the bash script for submitting it to CESGA Finis Terrae II Slurm queue system.  Additionally, folder contains a README_NoHooks.txt with a detailed explanation of the scripts.

LaunchServer_with_HOOK.

This example launches the Distributed TF server with the Yaroslav Bulatov solution. Both package modules (ServerDictionary and DistributedTFQueueHook) are needed. The example creates the server and the when the workers finished their job the "ps" server is closed and the job will be finished. Folder contains the python script and the bash script for submitting it to the queue system. Additionally, folder contains a README_WithHooks.txt with a detailed explanation of the scripts

MNIST_TUTORIAL

This Folder contains 4 different sub-folders with examples for deployment of a distributed Training of a Deep Convolutional Network using the MNIST dataset based on an example from TF webpage: mnist_deep.py file. Please follow the README.txt of the MNIST_TUTORIAL folder 


