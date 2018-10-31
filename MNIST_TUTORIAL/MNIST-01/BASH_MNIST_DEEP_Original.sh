#!/bin/sh

#SBATCH -N 1 # Number of Nodes
#SBATCH -n 1 # Number of tasks 
#SBATCH -c 1 # Number of cores per task
# SBATCH --ntasks-per-node=4 # para mpi: se solicitan 4 tareas por nodo

#SBATCH -t 00:10:00 #El tiempo maximo del trabajo es de 30 horas
# SBATCH --mem 30G

#SBATCH -p thinnodes
# SBATCH -p cola-corta
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
echo "TENSORFLOW: "$TENSORFLOW
PYTHON=$2
echo "PYTHON: "$PYTHON 
REDHAT=$(lsb_release -a | sed -n 's/Release:\t//p')
echo "REDHAT: "$REDHAT

# Limpio modulos
module purge

if [ $REDHAT = "6.7" ]
then
	MODULES=$(bash ../../tf4slurm/ModulesForRedHat6.7.sh $TENSORFLOW $PYTHON)

else
	MODULES=$(bash ../../tf4slurm/ModulesForRedHat7.5.sh $TENSORFLOW $PYTHON)
fi

echo "Here We go!!"
module load $MODULES

##########################################################################
#########For submitting LaunchTFServer.py to queue system ################
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
  MEMPERCORE=$(( $(sinfo -e -p $SLURM_JOB_PARTITION -o "%m/%c" -h) ))
fi
echo MEMPERCORE: $MEMPERCORE
MEMPERTASK=$(( $MEMPERCORE*$SLURM_CPUS_PER_TASK )) 
echo "RAM-PER-TASK: "$MEMPERTASK

MEMORY=10G
DATA_DIR=./MNIST_DATA_TF_$TENSORFLOW"_PYTHON_"$PYTHON
MODEL_DIR=./SINGLE_ORIGINAL_TF_$TENSORFLOW"_PYTHON_"$PYTHON
BATCH_SIZE=50
MAX_STEPS=1000
srun -n 1 -c $SLURM_CPUS_PER_TASK --mem $MEMORY python mnist_deep_modification.py --data_dir $DATA_DIR --model_dir $MODEL_DIR -batch_size $BATCH_SIZE -Iterations $MAX_STEPS > Log_Original_TF_$TENSORFLOW"_PYTHON_"$PYTHON".txt"

