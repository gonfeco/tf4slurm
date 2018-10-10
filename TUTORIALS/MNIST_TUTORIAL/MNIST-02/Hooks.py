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
