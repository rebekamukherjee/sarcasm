import copy
import random
import math
import numpy as np
import sys

#-------------------------
#  Extract features
#  from the LIBSVM format
#-------------------------
def extractfeatures(filename):
	f=open(filename,'r')
	data=f.read().strip()
	data=data.split('\n')
	for item in range(len(data)):
		data[item]=data[item].split(' ')
	
	maxval=[]
	invalid=[]
	for item in range(len(data)):
		if len(data[item])<3:
			invalid.append(item)
	n=0
	for i in invalid:
		data.pop(i-n)
		n+=1

	
	index=copy.deepcopy(data)

	for item in range(len(data)):		
		for i in range(1,len(data[item])):
			m=data[item][i].split(':')
			index[item][i]=m[0]
					
		index[item]=list(map(int,index[item]))
		if index[item][0]==-1:
			index[item][0]=0
		else:
			index[item][0]=1

		maxval.append(max(index[item]))
		
	maximum=max(maxval)
	return index,maximum
#-----------------------------------
# Training function
#-----------------------------------

def naivebayes(indices,r,length,lengthtrain):
	count_yes=0
	count_no=0
	length=length+1
	count_x1_y0=np.zeros(length)
	count_x1_y1=np.zeros(length)
	count_1=0
	count_2=0

	vocab_size=lengthtrain


	for item in range(len(indices)):
		if indices[item][0]==1:
			count_yes+=1
			for i in range(1,len(indices[item])):
				count_1+=1
				count_x1_y1[indices[item][i]]+=1
		elif indices[item][0]==0:
			count_no+=1
			for i in range(1,len(indices[item])):
				count_2+=1
				count_x1_y0[indices[item][i]]+=1

			
	
	prior_yes=math.log(((count_yes+r)/float(count_yes+count_no+2*r)))
	prior_no=math.log(((count_no+r)/float(count_yes+count_no+2*r)))	
	
	likelihood_x1_y1=np.zeros(length)
	likelihood_x0_y1=np.zeros(length)
	likelihood_x1_y0=np.zeros(length)
	likelihood_x0_y0=np.zeros(length)

	likelihood_x1_y1 = (count_x1_y1+r)/float(count_1 + vocab_size*r)
	likelihood_x0_y1=1-likelihood_x1_y1
	likelihood_x1_y0=(count_x1_y0+r)/float(count_2 + vocab_size*r)
	likelihood_x0_y0=1-likelihood_x1_y0
	likelihood_x1_y1=np.log(likelihood_x1_y1)
	likelihood_x0_y1=np.log(likelihood_x0_y1)
	likelihood_x1_y0=np.log(likelihood_x1_y0)
	likelihood_x0_y0=np.log(likelihood_x0_y0)


	return prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0, likelihood_x0_y0
				
#---------------------------
#  predict function
#---------------------------

def predict(indices,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,length):
	pos=0
	neg=0
	m=len(indices)
	netlikelihood_yes=np.full(m,prior_yes)
	netlikelihood_no=np.full(m,prior_no)

	for item in range(m):
		for i in range(length+1):
			if i in indices[item][1:]:
				netlikelihood_yes[item]+=likelihood_x1_y1[i]
				netlikelihood_no[item]+=likelihood_x1_y0[i]
			else:
				netlikelihood_yes[item]+=likelihood_x0_y1[i]
				netlikelihood_no[item]+=likelihood_x0_y0[i]
		
		if (netlikelihood_yes[item]) >= (netlikelihood_no[item]):
			if indices[item][0]==1:
				pos+=1
			else:
				neg+=1
		else:
			if indices[item][0]==0:
				pos+=1
			else:
				neg+=1

	accuracy=(pos/float(pos+neg))*100
	return accuracy


		
	
indextrain,lengthtrain=extractfeatures(sys.argv[6])
indextest,lengthtest=extractfeatures(sys.argv[7])

####################################CV SETS###########################
index0,length0=extractfeatures(sys.argv[1])
index1,length1=extractfeatures(sys.argv[2])
index2,length2=extractfeatures(sys.argv[3])
index3,length3=extractfeatures(sys.argv[4])
index4,length4=extractfeatures(sys.argv[5])


kfold0index=index0+index1+index2+index3
kfold1index=index1+index2+index3+index4
kfold2index=index0+index2+index3+index4
kfold3index=index0+index1+index3+index4
kfold4index=index0+index1+index2+index4


kfold1length=max(length1,length2,length3,length4)
kfold2length=max(length0,length2,length3,length4)
kfold3length=max(length0,length1,length3,length4)
kfold4length=max(length0,length1,length2,length4)
kfold0length=max(length0,length1,length2,length3)




kfoldlength=max(kfold0length,kfold1length,kfold2length,kfold3length,kfold4length)

#---------------------------------------------------
#   Cross Validation
#---------------------------------------------------
best=0

for r in [0.5,1,1.5,2]:
	prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(kfold1index,r,max(kfold1length,kfold0length),kfold1length)	
	acc0= predict(kfold0index,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(kfold1length,kfold0length))


	prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(kfold2index,r,max(kfold1length,kfold2length),kfold2length)	
	acc1= predict(kfold1index,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(kfold1length,kfold2length))


	prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(kfold3index,r,max(kfold3length,kfold2length),kfold3length)	
	acc2= predict(kfold2index,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(kfold2length,kfold3length))


	prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(kfold4index,r,max(kfold4length,kfold3length),kfold4length)	
	acc3= predict(kfold4index,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(kfold4length,kfold3length))


	prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(kfold0index,r,max(kfold4length,kfold0length),kfold0length)	
	acc4= predict(kfold4index,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(kfold4length,kfold0length))

	accuracy=(acc0+acc1+acc2+acc3+acc4)/5
	if accuracy>best:
		best=accuracy
		best_r=r




print ("Best cross validation accuracy:",accuracy)
print ("Best hyperparameter:",best_r)


#------------Training and testing--------------------

prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(indextrain,best_r,max(lengthtrain,lengthtest),lengthtrain)	
ac= predict(indextest,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(lengthtrain,lengthtest))

print ("Test accuracy:",ac)


#prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0=naivebayes(indextrain,best_r,max(lengthtrain,lengthtest),lengthtrain)	
acctrain= predict(indextrain,prior_yes,prior_no,likelihood_x1_y1,likelihood_x0_y1,likelihood_x1_y0,likelihood_x0_y0,max(lengthtrain,lengthtest))

print ("Training accuracy:",acctrain)















