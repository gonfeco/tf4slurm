# vim: ts=2
# This script creates a TensorFlow Distributed Server for Distributed Training
# of A deep MNIST classifier using convolutional layers (TF13)
# The scritp creates and deploys the Distributed TensorFlow server.
# The training is done by the workers and is calledi (line 60) from another file
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import time

import tensorflow as tf
sys.path.append("../../")
import tf4slurm.ServerDictionary as ServerDictionary
import tf4slurm.DistributedTFQueueHook as QueueHook
import DeepMNIST_DistributedTRAIN as TRAIN

FLAGS = None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir',type=str,default='./MNIST_DATA',help='Directory for storing input data. Default: ./MNIS_DATA')
	parser.add_argument('--model_dir',type=str,default='./MONITORED_DISTRIBUTED',help='Directory for storing input data. Default: ./MONITORED_DISTRIBUTED')
	parser.add_argument('-batch_size',type=int,default=50,help='Batch Size. Default: 50.')
	parser.add_argument('-Iterations',type=int,default=1005,help='Number of iterations. Default: 1000.')
	parser.add_argument('-ps', type=int, default=1, help='Number of Parameter Servers.')
	parser.add_argument('-workers', type=int, default=3, help='Number of Workers.')
	parser.add_argument('--NoIB', default=True,action="store_false", help='No use Infini Band.')

	FLAGS, unparsed = parser.parse_known_args()
	#Get Server Dictionary
	NumberOfPS=FLAGS.ps
	NumberOfWorkers=FLAGS.workers
	ListOfTFTasks=['worker' for i in range(NumberOfPS+NumberOfWorkers)]
	for i in range(NumberOfPS):
		ListOfTFTasks[i]='ps'
	ServerDictionary,task_type,task_index=ServerDictionary.GetServerDictionary(ListOfTFTasks,InfinyBand=FLAGS.NoIB)
	print("Cluster Dictionary: "+str(ServerDictionary))
	print(task_type,task_index)
	#Create TF Cluster
	cluster=tf.train.ClusterSpec(ServerDictionary)
	#Create TF Server
	server=tf.train.Server(cluster, job_name=task_type,task_index=task_index)
	print(server.target)
	if task_type == "ps":
		#Run Parameter Server
		queue=QueueHook.create_done_queue(task_index,NumberOfWorkers)
		session_ps=tf.Session(server.target)
		for i in range(NumberOfWorkers):
			session_ps.run(queue.dequeue())
			print("ps %d received done %d" % (task_index, i))
		session_ps.close()
	elif task_type  == "worker":
		QueueHook=QueueHook.QueueManagementHook(NumberOfPS=NumberOfPS,NumberOfWorkers=NumberOfWorkers)
		TRAIN.DistributedTrain(FLAGS, task_index, TFCluster=cluster, TFServer=server, QueueHook=QueueHook)
