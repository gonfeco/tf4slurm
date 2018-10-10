#!/bin/sh

#SBATCH -N 1 # Number of Nodes
#SBATCH -n 1 # Number of tasks 
#SBATCH -c 1 # Number of cores per task
# SBATCH --ntasks-per-node=4 # para mpi: se solicitan 4 tareas por nodo

#SBATCH -t 00:10:00 #El tiempo maximo del trabajo es de 30 horas
# SBATCH --mem 30G

#SBATCH -p cola-corta,thinnodes,iphinodes
# SBATCH -p thin-shared
# SBATCH -p gpunodes
# SBATCH -p iphinodes
# SBATCH -p fatnode
# SBATCH -p shared,gpu-shared-k2 --qos=shared
# SBATCH -p gpu-shared-k2 --qos=shared

##############################################################################
#To submit to the slurm: 
# sbatch BASH_MNIST_DEEP_Original.sh -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_MNIST_DEEP_Original.sh 10 -> use TensorFlow v1.0.0
# sbatch BASH_MNIST_DEEP_Original.sh 12 -> use TensorFlow v1.2.0
# sbatch BASH_MNIST_DEEP_Original.sh 13 -> use TensorFlow v1.3.0
# sbatch BASH_MNIST_DEEP_Original.sh 17 -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_MNIST_DEEP_Original.sh 17 2 -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_MNIST_DEEP_Original.sh 17 3 -> use TensorFlow v1.7.0 and Python 3.6.5
##############################################################################

TENSORFLOW=$1
PYTHON=$2

# Limpio modulos
module purge
case $TENSORFLOW in 
	10)
		module load gcc/4.9.1 tensorflow/1.0.0
	;;
	12)
		module load gcc/4.9.1 tensorflow/1.2.1
	;;
	13)
		module load gcc/4.9.1 tensorflow/1.3.1
	;;
	17)
	if [ "$PYTHON" = 2 ]
	then
		module load gcc/6.4.0 tensorflow/1.7.0-python-2.7.14
	elif [ "$PYTHON" = 3 ]
	then
		module load gcc/6.4.0 tensorflow/1.7.0-python-3.6.5
		export PYTHONPATH="/opt/cesga/job-scripts-examples/TensorFlow/Distributed:"$PYTHONPATH
	else
		echo "TF 17 will be used. Python Version not provided. Python 2 will be used"
		module load gcc/6.4.0 tensorflow/1.7.0-python-2.7.14

	fi
	;;
	*)
		echo "Not selected TensorFlow Version. v 1.7.0 and Python 2.7.14 will be used!!!"
		module load gcc/6.4.0 tensorflow/1.7.0-python-2.7.14
esac

##########################################################################
#########For submitting LaunchTFServer_Hooke.py to queue system ##########
##########################################################################

echo SLURM_NTASKS: $SLURM_NTASKS  
echo SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK 
echo SLURM_NNODES: $SLURM_NNODES
echo SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE

MEMORY=10G
DATA_DIR=./MNIST_DATA_TF_$TENSORFLOW"_PYTHON_"$PYTHON
MODEL_DIR=./SINGLE_ORIGINAL_TF_$TENSORFLOW"_PYTHON_"$PYTHON
BATCH_SIZE=50
MAX_STEPS=1000
srun -n 1 -c $SLURM_CPUS_PER_TASK --mem $MEMORY python mnist_deep_modification.py --data_dir $DATA_DIR --model_dir $MODEL_DIR -batch_size $BATCH_SIZE -Iterations $MAX_STEPS > Log_Original_TF_$TENSORFLOW"_PYTHON_"$PYTHON".txt"
