#!/bin/sh

#SBATCH -n 4 # Number of tasks 
#SBATCH -c 6 # Number of cores per task
#SBATCH --ntasks-per-node=4 # para mpi: se solicitan 4 tareas por nodo

#SBATCH -t 00:05:00 #El tiempo maximo del trabajo es de 30 horas

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
# sbatch BASH_LaunchTFServer_WithHOOK.sh -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_LaunchTFServer_WithHOOK.sh 10 -> use TensorFlow v1.0.0
# sbatch BASH_LaunchTFServer_WithHOOK.sh 12 -> use TensorFlow v1.2.0
# sbatch BASH_LaunchTFServer_WithHOOK.sh 13 -> use TensorFlow v1.3.0
# sbatch BASH_LaunchTFServer_WithHOOK.sh 17 -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_LaunchTFServer_WithHOOK.sh 17 2 -> use TensorFlow v1.7.0 and Python 2.7.14
# sbatch BASH_LaunchTFServer_WithHOOK.sh 17 3 -> use TensorFlow v1.7.0 and Python 3.6.5
##############################################################################

TENSORFLOW=$1
echo "TENSORFLOW: "$TENSORFLOW
PYTHON=$2
echo "PYTHON: "$PYTHON 
REDHAT=$(lsb_release -a | sed -n 's/Release:\t//p')
echo "REDHAT: "$REDHAT

# Limpio modulos
module purge

PATHTOPACKAGE="../tf4slurm"

if [ $REDHAT = "6.7" ]
then
	MODULES=$(bash $PATHTOPACKAGE/ModulesForRedHat6.7.sh $TENSORFLOW $PYTHON)

else
	MODULES=$(bash $PATHTOPACKAGE/ModulesForRedHat7.5.sh $TENSORFLOW $PYTHON)
fi

echo "Here We go!!"
module load $MODULES

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
  MEMPERCORE=$(( $(sinfo -e -p $SLURM_JOB_PARTITION -o "%m/%c" -h) ))
fi
echo MEMPERCORE: $MEMPERCORE
MEMPERTASK=$(( $MEMPERCORE*$SLURM_CPUS_PER_TASK )) 
echo "RAM-PER-TASK: "$MEMPERTASK

#Here The IPs of the nodes allocated for job are obtained and wrote to the environment variable TFSERVER
########################################################################################################
IB="NoIB"
export TFSERVER=""
if [ $IB = "NoIB" ]
  then
  #Get the IPs of all nodes of the allocated job. Ethernet IPs
  TFSERVER=$(srun -n $SLURM_NNODES --ntasks-per-node=1 $PATHTOPACKAGE/Wraper_NoIB.sh)
  else
  #Get the IPs of all nodes of the allocated job. Infiny Band IPs
  TFSERVER=$(srun -n $SLURM_NNODES --ntasks-per-node=1 $PATHTOPACKAGE/Wraper_IB.sh)
fi
echo $TFSERVER
########################################################################################################

#See https://github.com/tensorflow/models/issues/3788 to avoid
#tensorflow.python.framework.errors_impl.UnavailableError: OS Error
export GRPC_POLL_STRATEGY="poll"

PS=1
WORKERS=$((SLURM_NTASKS-PS))

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK  --mem $MEMPERTASK --resv-ports=$SLURM_NTASKS_PER_NODE -l python ./LaunchTFServer_Hook.py -ps $PS -workers $WORKERS 

