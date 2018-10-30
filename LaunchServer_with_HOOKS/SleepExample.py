# vim: ts=2
from __future__ import print_function
import tensorflow as tf
import time


def SleepExample(task_index, TFCluster, TFServer, QueueHook):
		is_chief=(task_index == 0)
		master=TFServer.target
		with tf.device(
			tf.train.replica_device_setter(
				worker_device="/job:worker/task:%d" %task_index,
				cluster=TFCluster,
			)
		):
			global_step=tf.train.get_or_create_global_step()
			#Here a increment operation over the global step is created
			increment_global_step_op=tf.assign(global_step, global_step+1)
		#Each worker will do 10 steps
		hooks=[QueueHook]
		hooks.append(tf.train.StopAtStepHook(last_step=10))
		Config=tf.ConfigProto(allow_soft_placement=True)#, log_device_placement=True)
		with tf.train.MonitoredTrainingSession(
			master=master,
			is_chief=is_chief,
			hooks=hooks,
			checkpoint_dir="./",
			save_checkpoint_secs=None,
			save_summaries_steps=None,
			config=Config
		) as sess:
			step=sess.run(global_step)
			while not sess.should_stop():
				#In order to have some time each step is delayed 10 seconds
				time.sleep(10)
				#Each step executes the increment operation on the global step.
				step=sess.run(increment_global_step_op)
