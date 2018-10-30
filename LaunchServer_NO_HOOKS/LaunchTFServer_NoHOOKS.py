# vim: ts=2
from __future__ import print_function
import argparse
import sys
import time
import os
import tensorflow as tf
sys.path.append("../")
import tf_for_slurm.ServerDictionary as ServerDictionary

###########################################################################
########This scripts shows an example of use of the tf_sfor_slurm##########
########Script creates a Distributed TF server and running it##############
###### Use server method join, so Server  running forever  ################
########Use LaunchTFServer.sh for submit this script to the queue##########
###########################################################################

if __name__ == "__main__":
	FLAGS=None
	parser=argparse.ArgumentParser()
	parser.add_argument('-ps', type=int, default=1, help='Number of Parameter Servers.')
	parser.add_argument('-workers', type=int, default=3, help='Number of Workers.')
	parser.add_argument('--NoIB', default=True,action="store_false", help='No use Infini Band.')

	FLAGS, unparsed = parser.parse_known_args()
	NumberOfPS=FLAGS.ps
	NumberOfWorkers=FLAGS.workers
	ListOfTFTasks=['worker' for i in range(NumberOfPS+NumberOfWorkers)]
	for i in range(NumberOfPS):
		ListOfTFTasks[i]='ps'
	ServerDictionary,task_type,task_index=ServerDictionary.GetServerDictionary(
		ListOfTFTasks,
		InfinyBand=FLAGS.NoIB
	)
	print("Cluster Dictionary: "+str(ServerDictionary))
	print(task_type,task_index)
	cluster = tf.train.ClusterSpec(ServerDictionary)
	server = tf.train.Server(
		cluster,
		job_name=task_type,
		task_index=task_index
	)
	print(server.target)
	print("I am "+task_type+" and my id inside the "+task_type+" list is: "+str(task_index)+ " and Network Address is: "+ServerDictionary[task_type][task_index])
	if task_type == "ps":
		server.join()
	elif task_type == "worker":
		pass
