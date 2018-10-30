# vim: ts=2
from __future__ import print_function
import argparse
import os
import tensorflow as tf
import sys
sys.path.append("../")
import tf4slurm.ServerDictionary as ServerDictionary
import tf4slurm.DistributedTFQueueHook as QueueHook
import SleepExample as SE
###########################################################################
########This scripts shows an example of use of the tf_sfor_slurm##########
########Script creates a Distributed TF server and running it##############
###### Use QueueManagement Hook so server will be closed #################
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
	ServerDictionary,task_type,task_index=ServerDictionary.GetServerDictionary(ListOfTFTasks,InfinyBand=FLAGS.NoIB)
	print("Cluster Dictionary: "+str(ServerDictionary))
	print(task_type,task_index)
	cluster = tf.train.ClusterSpec(ServerDictionary)
	server = tf.train.Server(cluster,
		job_name=task_type,
		task_index=task_index
	)
	print(server.target)
	print("I am "+task_type+" and my id inside the "+task_type+" list is: "+str(task_index)+ " and Network Address is: "+ServerDictionary[task_type][task_index])
	if task_type == "ps":
		queue=QueueHook.create_done_queue(task_index,FLAGS.workers)
		session_ps=tf.Session(server.target)
		for i in range(FLAGS.workers):
			session_ps.run(queue.dequeue())
			print("ps %d received done %d" % (task_index, i))
		session_ps.close()
	elif task_type  == "worker":
		QueueHook=QueueHook.QueueManagementHook(NumberOfPS=NumberOfPS,NumberOfWorkers=NumberOfWorkers)
		SE.SleepExample(task_index, TFCluster=cluster, TFServer=server, QueueHook=QueueHook)
