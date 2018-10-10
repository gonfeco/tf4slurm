# vim: ts=2
#import time
from __future__ import print_function
import tensorflow as tf

#####################################################################
#Hook class definition for closing Distributed TF server gracefully.
#https://gist.github.com/yaroslavvb/ea1b1bae0a75c4aae593df7eca72d9ca
#####################################################################

def create_done_queue(i,WorkersNumber):
	"""Queue used to signal death for i'th ps shard. Intended to have
	all workers enqueue an item onto it to signal doneness."""
	with tf.device("/job:ps/task:%d" % (i)):
		return tf.FIFOQueue(WorkersNumber, tf.int32, shared_name="done_queue"+
                        str(i))

def create_done_queues(PSNumber,WorkersNumber):
  return [create_done_queue(i,WorkersNumber) for i in range(PSNumber)]

class QueueManagementHook(tf.train.SessionRunHook):
	def __init__(self,NumberOfPS=None,NumberOfWorkers=None):#,chief=None):
		self.NumberOfPS=NumberOfPS
		self.NumberOfWorkers=NumberOfWorkers
		#self.chief=chief
		self.EnqueueOperations=[]
	def begin(self):
		#Needed for close ps server gracefully: YAROSLAB!!!
		for queue in create_done_queues(self.NumberOfPS,self.NumberOfWorkers):
			self.EnqueueOperations.append(queue.enqueue(1))

	def before_run(self, run_context):
		pass

	def after_run(self, run_context, run_values):
		pass

	def end(self, session):
		#if not self.chief:
		#	time.sleep(60)
		for QueueOperation in self.EnqueueOperations:
			session.run(QueueOperation)
		#session.close()
