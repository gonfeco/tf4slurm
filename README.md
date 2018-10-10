# tf4slurm
Integrating Distributed TensorFlow with Slurm queue systems
This Python Package was initially developed in Centro de Supercomputación de Galicia (CESGA, www.cesga.es) under the framework of the European Union’s H2020 research and innovation programme (grant agreement No 680481).
Complete description of this package and its successful application to an industrial case is provide in correspondent CESGA technical report: “Integrating Neural Network Parallel Training using Tensorflow with SLURM” (https://www.cesga.es/es/biblioteca/downloadAsset/id/803)
*****************************************************************************
Tree Folder Description:
There are 2 main folders in the package: the code (tf4slurm) and the examples (TUTORIALS).

**************************************************************************************************
tf4slurm
The tf_for_slurm folder contains the Python Package mandatory for using distributed Training
capabilities of TensorFlow in a Slurm queue system. There are two different modules in the package: ServerDictionary and DistributedTFQueueHook.

ServerDictionary
****************
All the basic functions to build the ServerDictionary for a Distributed TF Server are provided. When a Distributed TF training is required all the machines involved in the training need to load the "GetServerDictionary()" function of this script. 
This function needs two inputs:
  TensorFlowServerTasks: List of strings: Lenght of list = number of machines. Each element should be "ps" o "worker"
  InfiniBand: True (if infiniband is used). False (communincations through ethernet). This is a custom option for CESGA Finis       Terrae 2 system. You should check IP InfiniBand configuration of your system. 
The ouputs of the function are:
  DictionaryServer: python dictionary with the server configuration for the TF server: {"ps": [List of Addresses for "ps"],         "worker": [List of Addresses for "worker"]}. Same for all machines.
  task_type: string: "ps" or "worker". Depends of the machine.
  task_index: integer. Depends of the machine. Is the element of the machine in the list of the DictionaryServer depending of       the task_type.

DistributedTFQueueHook
**********************
This module defines the clase QueueManagementHook (based on tf.train.SessionRunHook) and the functions needed for closing gracefully the parameter server. This solution was based on the solution by Yaroslav Bulatov. (https://gist.github.com/yaroslavvb/ea1b1bae0a75c4aae593df7eca72d9ca).
This solution creates a TF Queues on all the parameter servers ("ps") and the "ps" session try to dequeue them. Meanwhile the Queue is empty this operation blocks the "ps" server. In the "worker" the Hook of the script is created and when the training operation is finished the hook fills the "ps" queues. When "ps" queue are filled the dequeue operation can be finished and the "ps" can be closed gracefully.
**************************************************************************************************
