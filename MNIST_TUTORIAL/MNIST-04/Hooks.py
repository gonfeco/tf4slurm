# vim: ts=2
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf

from datetime import datetime


class TBSummaryHook(tf.train.SessionRunHook):
	def __init__(self,SummaryOperation=None, output_dir=None,save_steps=None,max_steps=None):
		self.save_steps=save_steps
		self.TBwritter=tf.summary.FileWriter(output_dir)
		self.MergedSummaryOperation=SummaryOperation
		self.initial_step=None
		self.max_steps=max_steps
		self.should_trigger=False
	def begin(self):
		pass
	def after_create_session(self,session,coord):
		self.initial_step=session.run(tf.train.get_global_step())
	def before_run(self, run_context):
		global_step=run_context.session.run(tf.train.get_global_step())
		local_step=(global_step-self.initial_step)
		if local_step > self.save_steps:
			self.should_trigger=True
		else:
			self.should_trigger=False
		if self.should_trigger:
			self.initial_step=global_step
			return tf.train.SessionRunArgs(self.MergedSummaryOperation)
	def after_run(self, run_context, run_values):
		global_step=run_context.session.run(tf.train.get_global_step())
		if self.should_trigger:
			#print(run_values)
			self.TBwritter.add_summary(run_values.results,self.initial_step)
	def end(self, session):
		pass

class NewSummarySaverHook(tf.train.SessionRunHook):
	def __init__(self,SummaryOperation=None, output_dir=None,save_steps=None,max_steps=None,
		features=None,labels=None,dropout=None,batchx=None,batchy=None, dropout_value=None
	):
		self.save_steps=save_steps
		self.TBwritter=tf.summary.FileWriter(output_dir)
		self.MergedSummaryOperation=SummaryOperation
		self.features=features
		self.dropout=dropout
		self.labels=labels
		self.batchx=batchx
		self.batchy=batchy
		self.initial_step=None
		self.max_steps=max_steps
		self.should_trigger=False
		self.dropout_value=dropout_value
	#Before Create Session. Graph Modification allowed.
	def begin(self):
		pass
	def after_create_session(self,session,coord):
		global_step=session.run(tf.train.get_global_step())
		self.initial_step=global_step
		LocalSummary=session.run(
			self.MergedSummaryOperation,
			feed_dict={
				self.features:self.batchx,
				self.labels:self.batchy,
				self.dropout:self.dropout_value
			}
		)
		self.TBwritter.add_summary(LocalSummary,global_step)
	def before_run(self, run_context):
		pass
	def after_run(self, run_context, run_values):
		global_step=run_context.session.run(tf.train.get_global_step())
		self.should_trigger=((global_step-self.initial_step) > self.save_steps)
		if global_step >= self.max_steps:
			self.should_trigger=True

		if self.should_trigger:
			self.initial_step=global_step
			LocalSummary=run_context.session.run(
				self.MergedSummaryOperation,
				feed_dict={
					self.features:self.batchx,
					self.labels:self.batchy,
					self.dropout:self.dropout_value
				}
			)
			self.TBwritter.add_summary(LocalSummary,self.initial_step)
	def end(self, session):
		pass
		#global_step=session.run(tf.train.get_global_step())
		#LocalSummary=session.run(self.summary_op,feed_dict={self.features:self.batchx,self.labels:self.batchy})
		#self.TBwritter.add_summary(LocalSummary,global_step)

class FinalSummaryHook(tf.train.SessionRunHook):

	def __init__(self, SummaryDictionary, features, labels, keep_prob, batchx, batchy, prob, max_steps, Dataset):
		self.SummaryDictionary=SummaryDictionary
		self.features=features
		self.labels=labels
		self.keep_prob=keep_prob
		self.batchx=batchx
		self.batchy=batchy
		self.prob=prob
		self.max_steps=max_steps
		self.Dataset=Dataset

	def begin(self):
		pass

	def after_create_session(self,session,coord):
		pass

	def before_run(self, run_context):
		pass

	def after_run(self, run_context, run_values):
		global_step=run_context.session.run(tf.train.get_global_step())
		if global_step >= self.max_steps:
			FinalAcc=run_context.session.run(self.SummaryDictionary,feed_dict={self.features:self.batchx, self.labels:self.batchy, self.keep_prob:self.prob})
			print("Last step ("+str(global_step)+") Metrics on "+self.Dataset+" Dataset: ")
			for key in FinalAcc:
				print(key,FinalAcc[key])
			#for key, value in (FinalAcc).iteritems():
			#	print(key,value)

	def end(self, session):
		#LocalSummary=session.run(self.summary_op,feed_dict={self.features:self.batchx,self.labels:self.batchy})
		pass

######################################################################################################
# For synchronization of workers ate the beggining of the training.
######################################################################################################
class BarrierOnChiefHook(tf.train.SessionRunHook):
	def __init__(self,is_chief=None,chief_task_index=0,NumberOfWorkers=None):
		self._is_chief=is_chief
		self._chief_task_index=chief_task_index
		self._NumberOfWorkers=NumberOfWorkers
		self.ChiefQueue=None
		self.EnqueueOperations=[]
		self.DequeueOperations=[]
	def begin(self):
		#Need Create the Queue in the chief
		self.ChiefQueue=CreateQueue(self._chief_task_index, self._NumberOfWorkers-1)
		if self._is_chief:
			for queue in self.ChiefQueue:
				self.DequeueOperations.append(queue.dequeue())
		else:
			for queue in self.ChiefQueue:
				self.EnqueueOperations.append(queue.enqueue(1))
	def after_create_session(self,session,coord):
		if self._is_chief:
			for QueueOperation in self.DequeueOperations:
				session.run(QueueOperation )
		else:
			for QueueOperation in self.EnqueueOperations:
				session.run(QueueOperation)
		global_step=session.run(tf.train.get_global_step())
		#print("Just Start My step is: "+str(global_step))
		time.sleep(1)
	def before_run(self, run_context):
		pass
	def after_run(self, run_context, run_values):
		pass
	def end(self, session):
		pass

def CreateQueue(Worker, QueueSize):
	with tf.device("/job:worker/task:%d" % (Worker)):
		return [tf.FIFOQueue(QueueSize, tf.int32, shared_name="QueueOnWorker"+str(Worker))]

######################################################################################################
#For Monitoring training steps of each worker
######################################################################################################
class PrintStepHook():
	def __init__(self, AvoidStep=None):#,delay=60,is_chief=None):
		self._AvoidStep=AvoidStep
		pass
	def begin(self):
		pass
	def after_create_session(self,session,coord):
		pass
	def before_run(self, run_context):
		local_step=run_context.session.run(tf.train.get_global_step())
		if self._AvoidStep == None:
			print("I am step: "+str(local_step))
		else:
			if local_step < self._AvoidStep[0]:
				print("I am step: "+str(local_step)+" time is :"+str(datetime.now()))
			if local_step > self._AvoidStep[1]:
				print("I am step: "+str(local_step)+" time is :"+str(datetime.now()))
	def after_run(self, run_context, run_values):
		pass
	def end(self, session):
		print("I am in the End part of the Session and time is :"+str(datetime.now()))
