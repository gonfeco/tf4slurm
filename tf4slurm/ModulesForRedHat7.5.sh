#!/bin/sh

PYTHON=$2
TENSORFLOW=17

# Limpio modulos
module purge
case $TENSORFLOW in 
	10)
		MODULES="gcc/4.9.1 tensorflow/1.0.0"
	;;
	12)
		MODULES="gcc/4.9.1 tensorflow/1.2.1"
	;;
	13)
		MODULES="gcc/4.9.1 tensorflow/1.3.1"
	;;
	17)
	if [ "$PYTHON" = 2 ]
	then
		MODULES="gcc/6.4.0 tensorflow/1.7.0-python-2.7.15"
	elif [ "$PYTHON" = 3 ]
	then
		MODULES="gcc/6.4.0 tensorflow/1.7.0-python-3.7.0"
		#export PYTHONPATH="/opt/cesga/job-scripts-examples/TensorFlow/Distributed:"$PYTHONPATH
	else
		#echo "TF 17 will be used. Python Version not provided. Python 2 will be used"
		MODULES="gcc/6.4.0 tensorflow/1.7.0-python-2.7.15"

	fi
	;;
	*)
		#echo "Not selected TensorFlow Version. v 1.7.0 and Python 2.7.14 will be used!!!"
		MODULES="gcc/6.4.0 tensorflow/1.7.0-python-2.7.15"
esac
echo $MODULES
