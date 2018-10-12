# TUTORIAL: Deployment of Distributed Training of COnvolutional Network Training on MNIST database in FT2.

Here we developed an example of deployment of Ditributed TF using the TF webpage example: mnist_deep.py file.
We will do this in three steps in order to show user how to deploy distributed training in FT2. 
In this folder there are three sub-folders (MNIST-01, MNIST-02 and MNIST-3) that contains the different scripts. 
Each folder has its own Readme to explain all the major changes done in the code. The order of the folders is:

MNIST-01

*************************************************************************************************
Here we take the original TF Deep MNIST code from webpage and implement some minor modifications. 
Additionally a BASH script to submitt to the FT2 was created. 
If user use the script by default the Deep NN will be trained with the following parameters: 
	*BATCH_SIZE: 50, MAX_STEPS=1000. It will use only one core and wil reserve 10 G of RAM for the job. 
The results of the training will be stored in: Log_Original.txt and Training time should be near: 166.2 s.

MNIST-02:

*************************************************************************************************
Here we adapted the code from MNIST-01 to the use of monitoredTrainingSessions. Here we introduce too the TF Hooks. 
We split the code in two parts: a file where the Deep NN is created and a second file where the network is trained. 
In this Folder the training will be NOT DISTRIBUTED. A BASH script to submit the training to FT2 was created. 
If user use the script by default the Deep Network will be trained with the following parameters: 
	*BATCH_SIZE: 50, MAX_STEPS=1000. It will use only one core and wil reserve 10 G of RAM for the job. 
The results of the training will be stored in: Log_MonitoredSession.txt Training time should be near: 169.3 s. 
Here the trainign time is bigger than in MNIST-01 but we have more functionalities like monitoring of the training in TB.
Additionally important steps were done in order to achieve a future distributed Training deployment...

MNIST-03:

*************************************************************************************************
Here we adapt the code from MNIST-02 to final deployment in distributed. Again we split the code again. 
2 scripts were generated: one for created a Distributed TF server that calls to a second script
where the model is training in distributed way using the aforementioned Distributed server.
Here the adaptation of the code to distributed Trainign is straightofordward by taking the modified
code from MNIST-02 (see README_MNIST-03.txt). 
A BASH script to submit the training to FT2 was created. 
If user use the script by default the Deep NN will be trained with the following parameters: 
	*BATCH_SIZE: 50, MAX_STEPS=1000. It will use 4 tasks (1 core per task) for distributed Training. 
	Each task will reserve 10 G of RAM.
In this case the results are stored on Log_Distributed_0.txt that shows the log for all the tasks. Training time is near 112 s.
So with 3 workers we have passed from 166.2 to 112 so and speed up of 1.5 is achieved. 
User should be aware that this is a toy example so not big perofmances are expected. This is only a tutorial that shows
how users should adpat their code and how to use the tf_for_slurm package for deployment a TF training in Distributed way.

****************************************************************************************************************************
Additionally a last subfolder called MNIST-04 was created. In this script Distirbuted Synchronous training is implemented.



