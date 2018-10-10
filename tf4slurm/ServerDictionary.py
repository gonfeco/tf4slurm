# vim: set ts=2:
from __future__ import print_function
import os
import sys
PythonVersion=sys.version_info[0]
if (PythonVersion < 3) :
	import commands as commands
else:
	import subprocess as commands

##############################################################################
###This script contains all functions needed to build TF Dictionary Server####
##############################################################################


def GetServerDictionary(TensorFlowServerTasks,InfinyBand=True):
	"""
	This is the main function of the script:
	Inputs:
		TensorFlowServerTasks: list of string. Each element should be 'ps' or 'worker'.
		InfinyBand= Boolean. True->use InfinyBand. False->use Ethernet.
	Outputs:
		DictionaryServer: Server Dictionary for create Distributed TF server.
			keys:('ps', 'worker'). values: List with Network Addresses of the machines used for Distributed TF server.
		TypeOfTask: string.It can be: 'ps' or 'worker' depending of the type of TF task of the local machine.
		TypeOfTaskIndex:integer.Specify the index of the local machine in the Network Addresses List of its correspondent task.
	"""
	IPs,Ports,node_id,local_id=GetNodePortList(TensorFlowServerTasks,InfinyBand)
	#print("IPs: "+str(IPs))
	#print("Ports: "+str(Ports))
	#print("node_id: "+str(node_id))
	#print("local_id: "+str(local_id))

	#Create List combination with IPs and Ports.
	ListOfNodesPorts=CreateListNodesPorts(IPs,Ports)
	#print(ListOfNodesPorts)

	#Based on TensorFlowServerTasks create List for workers and for PSs
	ListForPS=[]
	ListForWorkers=[]
	for (index,task) in enumerate(TensorFlowServerTasks):
		if task == 'ps':
			ListForPS.append(ListOfNodesPorts[index])
		if task == 'worker':
			ListForWorkers.append(ListOfNodesPorts[index])
	#print("PS are : "+str(ListForPS))
	#print("Workers are: "+str(ListForWorkers))

	#Get the server Dictionary
	DictionaryServer={'ps':ListForPS,'worker':ListForWorkers}

	#Get the Machine for the local task.
	#This line is important: inside a node the tasks need to be associated to a communication port. Hence Posible cases:
	#1-> One node and several tasks: Each task will correspond with a Port of the ports list.
	#2-> Several Nodes one task/per_node:. In each node the task will correspond to the only communication port in port list.
	#3-> Several Nodes several tasks/per_node: similar to 1.
	local_task=IPs[node_id]+":"+Ports[local_id]
	#print("This task will be executed on machine: "+local_task)

	#Look machine of the local task in the PS List
	IsPS=local_task in ListForPS
	#print("Is a PS: "+str(IsPS))
	if IsPS:
		TypeOfTaskIndex=ListForPS.index(local_task)
		TypeOfTask='ps'

	#Look machine of the local task in the Worker List
	IsWorker=local_task in ListForWorkers
	#print("Is a Worker: "+str(IsWorker))
	if IsWorker:
		TypeOfTaskIndex=ListForWorkers.index(local_task)
		TypeOfTask='worker'

	#print(DictionaryServer)
	#print("TypeOfTask: "+TypeOfTask)
	#print("TypeOfTaskIndex: "+str(TypeOfTaskIndex))
	return DictionaryServer,TypeOfTask,TypeOfTaskIndex


def GetNodePortList(TensorFlowServerTasks,InfinyBand=True):
	"""
	Read From Environment SLURM_STEP_NODELIST and SLURM_STEP_RESV_PORTS
	Input:
		TensorFlowServerTasks: List with server tasks (ps or worker). One to one Correlation with ListOfNodes
		InfinyBand: True for get IP of InfinyBand
	Outputs:
		IPs:List with IPs reserved.
		Ports:List with Ports reserved.
		node_id: Index of the node for the current task.
		local_id: Node Local task Id for current task.
	"""
	NumberOfTasks=int(LookInEnvironment("SLURM_NTASKS"))
	if NumberOfTasks > len(TensorFlowServerTasks):
		print("Number of SLURM_NTASKS bigger than NumberOfPS+NumberOfWorkers")
		raise ValueError
	elif NumberOfTasks < len(TensorFlowServerTasks):
		print("Number of SLURM_NTASKS lower than NumberOfPS+NumberOfWorkers")
		raise ValueError
	else:
		pass

	#Get Nodes
	NODESEnv=LookInEnvironment("SLURM_STEP_NODELIST")
	#ExitStatus,SNodes=commands.getstatusoutput("nodeset -e -S, "+NODESEnv)
	ExitStatus,SNodes=commands.getstatusoutput("PYTHONPATH="" nodeset -e -S, "+NODESEnv)
	if (ExitStatus != 0):
		#print("Problem with SLURM_JOB_NODELIST format!.\n See man nodeset to obtain format!")
		print("Problem with SLURM_STEP_NODELIST format!.\n See man nodeset to obtain format!")
		raise ValueError
	IPs=SNodes.split(',')
	#print IPs
	#Get Ifiny Band IP
	if (InfinyBand == True):
		IPs=GetInfiniBandIp(IPs)
	#print IPs

	#Get Ports
	PORTSEnv=LookInEnvironment("SLURM_STEP_RESV_PORTS")
	#ExitStatus,SPorts=commands.getstatusoutput("nodeset -e -S, "+"p["+PORTSEnv+"]")
	ExitStatus,SPorts=commands.getstatusoutput("PYTHONPATH="" nodeset -e -S, "+"p["+PORTSEnv+"]")
	#print(SPorts)
	if (ExitStatus != 0):
		print("Problem with SLURM_STEP_RESV_PORTS format!.\n Accepted formats: Port0-Portf, Port0,Port1,Port2, Port0")
		raise ValueError
	#print "SPorts= "+SPorts
	Ports=(SPorts.replace('p','')).split(',')
	#task_id=int(LookInEnvironment("SLURM_PROCID"))
	node_id=int(LookInEnvironment("SLURM_NODEID"))
	local_id=int(LookInEnvironment("SLURM_LOCALID"))
	#print(node_id, local_id)
	return IPs, Ports, node_id, local_id

def LookInEnvironment(EnvVar):
	"""
	This function tryes to evaluate all environment varible passed to it.
	Inputs:
		EnvVar: Environmnet variable to evalutate its value.
	Outputs:
		EnvValue: Environment variable value (if properly read)
	"""
	try:
		EnvValue=os.environ[EnvVar]
		#print flag
	except KeyError:
		print("Not "+EnvVar+" environment variable!!")
		raise KeyError
	return EnvValue

def GetInfiniBandIp(IPs, IB0="172.28."):
	"""
	Get List Of Ips of InfinyBand for list of Nodes.
	Input:
		IPs:List Of reserved Nodes.
	Otuputs:
		InfinyBandIPs: List of Correspondient InfinyBand Ips
	WARNING!!! 
	This is only for CESGA Finis Terrae II Slurm system. In order to use in other systems you need
	to check you IP InfiniBand system!!!!
	"""
	#IB0="172.28."
	InfinyBandIPs=[]
	for ip in IPs:
		ip0=int(ip[1:3])
		ip1=int(ip[3:])
		InfinyBandIPs.append(IB0+str(ip0)+"."+str(ip1))
	return InfinyBandIPs

def CreateListNodesPorts(IPs,Ports):
	ListOfNodesPorts=[]
	#ListOfPorts=[]
	for node in IPs:
		for prts in Ports:
			#ListOfNodes.append(node)
			#ListOfPorts.append(prts)
			ListOfNodesPorts.append(node+":"+prts)
	#return ListOfNodes,ListOfPorts
	return ListOfNodesPorts

