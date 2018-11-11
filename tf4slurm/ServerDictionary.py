# vim: set ts=2:
from __future__ import print_function
import os
import sys
from time import sleep
PythonVersion=sys.version_info[0]
if (PythonVersion < 3) :
	import commands as commands
else:
	import subprocess as commands

##############################################################################
###This script contains all functions needed to build TF Dictionary Server####
##############################################################################


def LookInEnvironment(EnvVar):
	"""
	Esta funcion intentan evaluar la variable de entorno que se le pasa.
	Inputs:
		EnvVar: Variable de entorno de la que se quiere saber el valor. Si no la hay levanta un error.
	Outputs:
		EnvValue: Valor de la variable de entorno si se consigue leer correctamente.
	"""
	try:
		EnvValue=os.environ[EnvVar]
		#print flag
	except KeyError:
		print("Not "+EnvVar+" environment variable!!")
		raise KeyError
	return EnvValue

def GetInfiniBandIp(IPs):
	"""
	Get List Of Ips of InfinyBand for list of Nodes.
	Input:
		IPs:List Of reserved Nodes.
	Otuputs:
		InfinyBandIPs: List of Correspondient InfinyBand Ips
	"""
	IB0="172.28."
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

def TestNumberOfTasks(TensorFlowServerTasks):
	"""
	Check If Number of Tasks readed from Environment Variable is the same 
	that the lengt of list of TensorFlowServerTasks. If not raise an exception
	Input:
		TensorFlowServerTasks: List with server tasks (ps or worker). One to one Correlation with ListOfNodes
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
	return 0

"""
def GetNodesIPs():
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
	return IPs
"""

def GetNodesIPs():
	#Get Nodes
	NODESEnv=LookInEnvironment("TFSERVER")
	#print(NODESEnv)
	NodesIPs=NODESEnv.split('\n')
	#print(NodesIPs)
	return NodesIPs

def GetCommunicationPorts(): 
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
	return Ports

def CreateDictionary4Server(TensorFlowServerTasks,ListOfNodesPorts):
	"""
	Create Python Dictionary for Distributed TF Server
	Inputs:
		TensorFlowServerTasks: list of string. Each element should be 'ps' or 'worker'.
		ListOfNodesPorts: list of available IPs:CommunicationPort for all the tasks of thei future Distributed TF server.
	Outputs:
		DictionaryServer: Python Dictionary with the correct format to create Distributed TF server.
	"""
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
	#print(DictionaryServer)
	return DictionaryServer

def GetServerDictionary(ListOfTFTasks):
	"""
	This is the WorkFlow for the script:
	Inputs:
		ListOfTFTasks: list of string. Each element should be 'ps' or 'worker'.
	Outputs:
		DictionaryServer: Server Dictionary for create Distributed TF server.
			keys:('ps', 'worker'). values: List with Network Addresses of the machines used for Distributed TF server.
		TypeOfTask: string.It can be: 'ps' or 'worker' depending of the type of TF task of the local machine.
		TypeOfTaskIndex:integer.Specify the index of the local machine in the Network Addresses List of its correspondent task.
	"""
	print("I am new version with getting IPs!!!")
	#Test Number of Tasks
	TestNumberOfTasks(ListOfTFTasks)
	#Get List with Nodes IP
	ListOfIPs=GetNodesIPs()
	#Get List with Communication Ports
	ListOfPorts=GetCommunicationPorts()
	#Combine Ips and Ports with correct Distributed TF server.
	ListOfNodesPorts=CreateListNodesPorts(ListOfIPs,ListOfPorts)
	#Create the Python Dictionary Needed for Distributed TF server
	DictionaryServer=CreateDictionary4Server(ListOfTFTasks,ListOfNodesPorts)
	#Identification of current task
	node_id=int(LookInEnvironment("SLURM_NODEID"))
	local_id=int(LookInEnvironment("SLURM_LOCALID"))
	#Get the Machine for the local task.
	#This line is important: inside a node the tasks need to be associated to a communication port. Hence Posible cases:
	#1-> One node and several tasks: Each task will correspond with a Port of the ports list.
	#2-> Several Nodes one task/per_node:. In each node the task will correspond to the only communication port in port list.
	#3-> Several Nodes several tasks/per_node: similar to 1.
	local_task=ListOfIPs[node_id]+":"+ListOfPorts[local_id]
	#print("This task will be executed on machine: "+local_task)
	#Look machine of the local task in the PS List
	ListForPS=DictionaryServer['ps']
	IsPS=local_task in ListForPS
	#print("Is a PS: "+str(IsPS))
	if IsPS:
		TypeOfTaskIndex=ListForPS.index(local_task)
		TypeOfTask='ps'
	#Look machine of the local task in the Worker List
	ListForWorkers=DictionaryServer['worker']
	IsWorker=local_task in ListForWorkers
	#print("Is a Worker: "+str(IsWorker))
	if IsWorker:
		TypeOfTaskIndex=ListForWorkers.index(local_task)
		TypeOfTask='worker'
	return DictionaryServer,TypeOfTask,TypeOfTaskIndex

