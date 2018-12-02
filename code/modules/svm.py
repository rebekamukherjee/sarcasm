import copy
import random
import numpy as np
import sys


count20=[0]*20


#----------------------------------------------
# Extract features from the LIBSVM format files
#----------------------------------------------

def extractfeatures(filename):
	f=open(filename,'r')
	data=f.read().strip()
	data=data.split('\n')
	for item in range(len(data)):
		data[item]=data[item].split(' ')
	a=copy.deepcopy(data)
	index=copy.deepcopy(data)
	value=copy.deepcopy(data)
	maxval=[]
	invalid=[]
	for item in range(len(data)):
		if len(data[item])<3:
			invalid.append(item)
	n=0
	for i in invalid:
		data.pop(i-n)
		n+=1

	a=copy.deepcopy(data)
	index=copy.deepcopy(data)
	value=copy.deepcopy(data)
	

	for item in range(len(data)):
		a[item]=a[item][0:1]
		t=data[item]
		if data[item][0]=="1":
			a[item][0]=1
		else:
			a[item][0]=-1
		
		
		for i in range(1,len(data[item])):
			m=data[item][i].split(':')
			index[item][i]=m[0]
			value[item][i]=m[1]
			
		
		index[item]=index[item][1:]
		value[item]=value[item][1:]
		index[item]=list(map(int,index[item]))
		value[item]=list(map(float,value[item]))
		a[item]=list(map(int,a[item]))
		index[item].insert(0,0)
		value[item].insert(0,1)	
		maxval.append(max(index[item]))

		
	maximum=max(maxval)

	return index,value,a,maximum



#############################  CV sets making ##########################################	

index0,value0,data0,length0=extractfeatures(sys.argv[1])
index1,value1,data1,length1=extractfeatures(sys.argv[2])
index2,value2,data2,length2=extractfeatures(sys.argv[3])
index3,value3,data3,length3=extractfeatures(sys.argv[4])
index4,value4,data4,length4=extractfeatures(sys.argv[5])

kfold0index=index0+index1+index2+index3
kfold1index=index1+index2+index3+index4
kfold2index=index0+index2+index3+index4
kfold3index=index0+index1+index3+index4
kfold4index=index0+index1+index2+index4

kfold0value=value0+value1+value2+value3
kfold1value=value1+value2+value3+value4
kfold2value=value0+value2+value3+value4
kfold3value=value0+value1+value3+value4
kfold4value=value0+value1+value2+value4

kfold0data=data0+data1+data2+data3
kfold1data=data1+data2+data3+data4
kfold2data=data0+data2+data3+data4
kfold3data=data0+data1+data3+data4
kfold4data=data0+data1+data2+data4


kfold1length=max(length1,length2,length3,length4)
kfold2length=max(length0,length2,length3,length4)
kfold3length=max(length0,length1,length3,length4)
kfold4length=max(length0,length1,length2,length4)
kfold0length=max(length0,length1,length2,length3)

kfoldlength=max(kfold0length,kfold1length,kfold2length,kfold3length,kfold4length)

#######################  training function ##################################

def SVM(index,value,data,length,r0,C):
	random.seed(40)
	w=np.random.uniform(-0.01,0.01,length+2)

	b=random.uniform(-0.01,0.01)

	product=copy.deepcopy(index)
	best=copy.deepcopy(index)
	predicted=0
	for item in range(len(product)):
		product[item]=product[item][:1]
		product[item][0]=0
	t=0
	for epoch in range(0,10):
		r=r0/(1+((r0*t)/C))
		for item in range(len(index)):
			product[item][0]=0
			label=data[item][0]
			for i in range(len(index[item])):
				k=index[item][i]
				product[item][0]+=w[k]*value[item][i]
			predicted=product[item][0]+b
			
			if predicted*label<=1:
				w=w*(1-r)
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=(w[k]) +(r* C* label*(value[item][i]))
					
				b=b*(1-r)+ r *C*label
				
			else:
				w=w*(1-r)
				b=b*(1-r)
			predicted=0	
			
		best[epoch]=w	
		c=list(zip(index,value,data))
		random.shuffle(c)
		index,value,data=list(zip(*c))
		t+=1
		
		
	return w,b





###########################test function#######################################



def predict(weight,bias,data,index,value):
	product=copy.deepcopy(index)
	positive=0
	negative=0
	predicted=0
	for item in range(len(product)):
		product[item]=product[item][:1]
		product[item][0]=0
	
	for item in range(len(data)):
		label=data[item][0]
		for i in range(len(index[item])):
			k=index[item][i]
			product[item][0]+=weight[k]*value[item][i]
		predicted=product[item][0]+bias
		
		if predicted >= 0:
			predictedlabel=1
		else:
			predictedlabel=-1
		if predictedlabel==label:
			positive=positive+1
		else:
			negative=negative+1
		
	accuracy=((positive)/float(positive+negative))*100
	return accuracy


################Average Function################################
def average(number1,number2,number3,number4,number5):
	number=number1+number2+number3+number4+number5	
	average=number/5
	return average


######################cross validation############ uncomment this part to check for the best CV hyperparameters################
best_val=0

for r0 in [10,1,0.1,0.01,0.001,0.0001]:
	for C in [10,1,0.1,0.01,0.001,0.0001]:
		
		#print '========================================='
		#print 'Learning rate' , r0, 'Loss parameter', C
		#print '========================================='
		weight0,bias0=SVM(kfold1index,kfold1value,kfold1data,kfoldlength,r0,C)
		weight1,bias1=SVM(kfold2index,kfold2value,kfold2data,kfoldlength,r0,C)
		weight2,bias2=SVM(kfold3index,kfold3value,kfold3data,kfoldlength,r0,C)
		weight3,bias3=SVM(kfold4index,kfold4value,kfold4data,kfoldlength,r0,C)
		weight4,bias4=SVM(kfold0index,kfold0value,kfold0data,kfoldlength,r0,C)

		predict0=predict(weight0,bias0,data0,index0,value0)
		predict1=predict(weight1,bias1,data1,index1,value1)
		predict2=predict(weight2,bias2,data2,index2,value2)
		predict3=predict(weight3,bias3,data3,index3,value3)
		predict4=predict(weight4,bias4,data4,index4,value4)

		averageval=average(predict0,predict1,predict2,predict3,predict4)
		#print 'Average accuracy of 5 folds is :',averageval ,'%'
		if averageval>best_val:
			best_val=averageval
			best_r=r0
			best_C=C

print ('Best cross validation accuracy',best_val,'with learning rate', best_r,'loss parameter', best_C)


#################train for 20 epochs######################################
def SVM_1(index,value,data,length,r0,C):
	global count20
	random.seed(40)
	w=np.random.uniform(-0.01,0.01,length+2)
	
	b=random.uniform(-0.01,0.01)
	
	product=copy.deepcopy(index)
	bestweight=copy.deepcopy(index)
	bestbias=[]
	predicted=0
	for item in range(len(product)):
		product[item]=product[item][:1]
		product[item][0]=0
	
	t=0

	for epoch in range(0,20):
		r=r0/(1+((r0*t)/C))
		for item in range(len(index)):
			product[item][0]=0
			label=data[item][0]
			for i in range(len(index[item])):
				k=index[item][i]
				product[item][0]+=w[k]*value[item][i]
			predicted=product[item][0]+b
			
			if predicted*label<=1:
				w=w*(1-r)
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=(w[k]) +(r* C* label*(value[item][i]))
					
				b=b*(1-r)+ r *C*label
				
			else:
				w=w*(1-r)
				b=b*(1-r)
				
			predicted=0
			
		bestweight[epoch]=w
		bestbias.append(b)	
		c=list(zip(index,value,data))
		random.shuffle(c)
		index,value,data=list(zip(*c))
		t+=1
		
		
	return bestweight,bestbias

############ Testing  ###################
indextest,valuetest,datatest,lengthtest=extractfeatures(sys.argv[7])
indextrain, valuetrain, datatrain, lengthtrain=extractfeatures(sys.argv[6])
weighttrain,biastrain=SVM_1(indextrain,valuetrain,datatrain,max(lengthtest,lengthtrain),best_r,best_C)

predict_test=copy.deepcopy(biastrain)

for epoch in range(len(biastrain)):
	predict_test[epoch]=predict(weighttrain[epoch],biastrain[epoch],datatest,indextest,valuetest)

predictaccuval=predict(weighttrain[19],biastrain[19],datatest,indextest,valuetest)
predicttrain1=predict(weighttrain[19],biastrain[19],datatrain,indextrain,valuetrain)
#############best epoch finding function##########################
def findepoch(test):
	maxval=max(test)
	bestepoch=test.index(maxval)
	return maxval, bestepoch
	
#######################################
bestaccu,bestepoch=findepoch(predict_test)
print ('Best value of test set accuracy:',bestaccu,'%', 'across epoch:', bestepoch)
print ('Training accuracy:',predicttrain1, '%')
print ('Overall test set accuracy' , predictaccuval ,'%')

