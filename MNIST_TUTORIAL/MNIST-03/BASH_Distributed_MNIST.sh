#!/bin/sh

#SBATCH -N 1 # Number of Nodes
#SBATCH -n 4 # Number of tasks 
#SBATCH -c 1 # Number of cores per task
#SBATCH --ntasks-per-node=4 # para mpi: se solicitan 4 tareas por nodo

#SBATCH -t 00:10:00 #El tiempo maximo del trabajo es de 30 horas

#SBATCH -p cola-corta,thinnodes
# SBATCH -p thin-shared
# SBATCH -p gpunodes
# SBATCH -p iphinodes
# SBATCH -p fatnode
# SBATCH -p shared,gpu-shared-k2 --qos=shared
# SBATCH -p gpu-shared-k2 --qos=shared

##############################################################################
#To submit to the slurm: 
# sbatch BASH_Distributed_MNIST.sh -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_Distributed_MNIST.sh 10 -> use TensorFlow v1.0.0
# sbatch BASH_Distributed_MNIST.sh 12 -> use TensorFlow v1.2.0
# sbatch BASH_Distributed_MNIST.sh 13 -> use TensorFlow v1.3.0
# sbatch BASH_Distributed_MNIST.sh 17 -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_Distributed_MNIST.sh 17 2 -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_Distributed_MNIST.sh 17 3 -> use TensorFlow v1.7.0 and Python 3.6.5
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
#Memory Calculations.
#For shared partitions
MEMPERCORE=$(eval $(scontrol show partition $SLURM_JOB_PARTITION -o);echo $DefMemPerCPU)
if [ -z "$MEMPERCORE" ]
  then
  #exclusive partitions
  MEMPERCORE=$(( $(sinfo -e -p $SLURM_JOB_PARTITION -N -o "%m/%c" -h) ))
fi
echo MEMPERCORE: $MEMPERCORE
MEMPERTASK=$(( $MEMPERCORE*$SLURM_CPUS_PER_TASK )) 
echo "RAM-PER-TASK: "$MEMPERTASK

PS=1
WORKERS=$((SLURM_NTASKS-PS))

#MEMORY=$MEMPERTASK
MEMORY=10G
DATA_DIR=./MNIST_DATA_TF_$TENSORFLOW"_PYTHON_"$PYTHON
MODEL_DIR=./DISTRIBUTED_MONITORED_TF_$TENSORFLOW"_PYTHON_"$PYTHON
BATCH_SIZE=50
MAX_STEPS=1000

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK  --mem $MEMORY  --resv-ports=$SLURM_NTASKS_PER_NODE -l python LaunchTFServerWithHooks.py -ps $PS -workers $WORKERS --data_dir $DATA_DIR --model_dir $MODEL_DIR -batch_size $BATCH_SIZE -Iterations $MAX_STEPS --NoIB 


