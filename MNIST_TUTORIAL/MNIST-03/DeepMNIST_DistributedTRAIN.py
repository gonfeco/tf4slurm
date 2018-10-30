# vim: ts=2
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#			http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import Model_For_mnist_deep as Deep
import Hooks as HOOKS
from tensorflow.examples.tutorials.mnist import input_data

def DistributedTrain(FLAGS, task_index, TFCluster, TFServer,QueueHook):
	is_chief=(task_index == 0)
	master=TFServer.target
	with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" %task_index,cluster=TFCluster)):
		#Import data
		mnist=input_data.read_data_sets(FLAGS.data_dir+"-"+str(task_index), one_hot=True)
		#PlaceHolder for Images
		x = tf.placeholder(tf.float32, [None, 784])
		#PlaceHolder for Labels-OneHot Vector
		y_ = tf.placeholder(tf.float32, [None, 10])
		#Build the Deep Model
		y_conv, keep_prob = Deep.deepnn(x)
		#Define loss and optimizer
		SummaryDictionary={}
		global_step=tf.train.get_or_create_global_step()
		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
		cross_entropy = tf.reduce_mean(cross_entropy)
		SummaryDictionary.update({'cross_entropy': cross_entropy})
		with tf.name_scope('adam_optimizer'):
			Optimizer=tf.train.AdamOptimizer(1e-4)
			train_step=Optimizer.minimize(cross_entropy,global_step=global_step)
		#Define Metric
		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)
		SummaryDictionary.update({'AccuracyMetric': accuracy})

	#Added Summaries for TensorBoard Visualization.
	ListOfSummaries=[]
	for key in SummaryDictionary:
		ListOfSummaries.append(tf.summary.scalar(key,SummaryDictionary[key]))
	#for key, value in (SummaryDictionary).iteritems():
	#	ListOfSummaries.append(tf.summary.scalar(key,value))
	MergedSummaryOperation=tf.summary.merge(ListOfSummaries)

	BatchSize=FLAGS.batch_size
	SummarySteps=100
	hooks=[QueueHook]
	#Hook for StopAtStepHook
	hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.Iterations))
	if is_chief:
		#Get training subset for validation
		TrainingLenght=mnist.train.images.shape[0]
		#idx=np.random.randint(TrainingLenght,size=int(TrainingLenght/10))
		idx=np.random.randint(TrainingLenght,size=BatchSize)
		TrainingImagesValidation=mnist.train.images[idx]
		#print(TrainingImagesValidation.shape)
		TrainingLabelsValidation=mnist.train.labels[idx]
		#print(TrainingLabelsValidation.shape)
		hooks.append(HOOKS.NewSummarySaverHook(MergedSummaryOperation, FLAGS.model_dir+"/Training", SummarySteps, FLAGS.Iterations,
			features=x,labels=y_,dropout=keep_prob,batchx=TrainingImagesValidation, batchy=TrainingLabelsValidation, dropout_value=1.0)
		)
	hooks.append(HOOKS.FinalSummaryHook(SummaryDictionary,x, y_, keep_prob, mnist.test.images, mnist.test.labels, 1.0, FLAGS.Iterations,"Testing"))
	tick=time.time()
	#hooks.append(HOOKS.BarrierOnChiefHook(is_chief=is_chief,chief_task_index=0,NumberOfWorkers=FLAGS.workers))
	Config=None
	with tf.train.MonitoredTrainingSession(master=master,is_chief=is_chief,hooks=hooks,
		checkpoint_dir=FLAGS.model_dir,save_checkpoint_secs=None,save_summaries_steps=None,save_summaries_secs=None,
		config=Config
	) as sess:
		step=sess.run(global_step)
		while not sess.should_stop(): #and step < FLAGS.Iterations:
			batch=mnist.train.next_batch(BatchSize)
			_,step=sess.run([train_step,global_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			tack=time.time()
	print("Training Time: "+str(tack-tick))
