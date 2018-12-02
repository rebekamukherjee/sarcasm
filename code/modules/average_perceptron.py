import copy
import random
import matplotlib.pyplot as plt
import sys

count20=[0]*20

#-------------------------------
# Extract features 
# from the LIBSVM format files
#-------------------------------

def extractfeatures(filename):
	f=open(filename,'r')
	data=f.read().strip()
	data=data.split('\n')
	#print data
	for item in range(len(data)):
		data[item]=data[item].split(' ')
	a=copy.deepcopy(data)
	index=copy.deepcopy(data)
	value=copy.deepcopy(data)
	maxval=[]
	for item in range(len(data)):
		a[item]=a[item][0:1]
		t=data[item]
		if data[item][0]=="1":
			a[item][0]=1
		else:
			a[item][0]=0
		
		for i in range(1,len(data[item])):
			m=data[item][i].split(':')
			index[item][i]=m[0]
			value[item][i]=m[1]
		index[item]=index[item][1:]
		value[item]=value[item][1:]
		index[item]=map(int,index[item])
		value[item]=map(float,value[item])
		a[item]=map(int,a[item])	
		#print index[item]
		#print value[item]
		#print max(index[item])
		maxval.append(max(index[item]))
		
		#print data[item][1]
		#print '\n'
	maximum=max(maxval)
	#print maximum

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

def perceptron(index,value,data,length,r0):
	random.seed(40)
	w=[0.005]*(length+1)
	for val in range(len(w)):
		w[val]=random.uniform(-0.01,0.01)
	a=[0.005]*(length+1)
	for val in range(len(a)):
		a[val]=random.uniform(-0.01,0.01)

	b=random.uniform(-0.01,0.01)
	ba=random.uniform(-0.01,0.01)
	product=copy.deepcopy(index)
	predicted=0
	for item in range(len(product)):
		product[item]=product[item][:1]
		product[item][0]=0
		#print product[item]
	t=0
	for epoch in range(0,10):
		r=r0/1+t
		for item in range(len(index)):
			product[item][0]=0
			label=data[item][0]
			for i in range(len(index[item])):
				k=index[item][i]
				#print k
				product[item][0]+=w[k]*value[item][i]
			predicted=product[item][0]+b
			if predicted >= 0:
				predictedlabel=1
			else:
				predictedlabel=0
			if predictedlabel==1 and label==0:
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=w[k]-r*(value[item][i])
				b=b-r
			elif predictedlabel==0 and label==1:
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=w[k]+r*(value[item][i])
				b=b+r
			else:
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=w[k]		
				b=b
		for val in range(len(a)):
			a[val]=a[val]+w[val]
		ba=ba+b			
		
	
		c=list(zip(index,value,data))
		random.shuffle(c)
		index,value,data=list(zip(*c))
		t+=1
		
		#print w
	#print w
	#print b
	return a,ba
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
		#print label
		for i in range(len(index[item])):
			k=index[item][i]
			product[item][0]+=weight[k]*value[item][i]
		predicted=product[item][0]+bias
		if predicted >= 0:
			predictedlabel=1
		else:
			predictedlabel=0
		#print predictedlabel
		if predictedlabel==label:
			positive=positive+1
		else:
			negative=negative+1
		
	accuracy=((positive)/float(positive+negative))*100
	#print 'accuracy :' ,accuracy
	return accuracy



##########################################################################
def average(number1,number2,number3,number4,number5):
	number=number1+number2+number3+number4+number5	
	average=number/5
	return average

######################call perceptron############################

best_val=0

for r0 in [1,0.1,0.01,0.001,0.0001]:
		
		#print '========================================='
		#print 'Learning rate' , r0, 'Loss parameter', C
		#print '========================================='
		weight0,bias0=perceptron(kfold1index,kfold1value,kfold1data,kfoldlength,r0)
		weight1,bias1=perceptron(kfold2index,kfold2value,kfold2data,kfoldlength,r0)
		weight2,bias2=perceptron(kfold3index,kfold3value,kfold3data,kfoldlength,r0)
		weight3,bias3=perceptron(kfold4index,kfold4value,kfold4data,kfoldlength,r0)
		weight4,bias4=perceptron(kfold0index,kfold0value,kfold0data,kfoldlength,r0)

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


print ('Best cross validation accuracy',best_val,'with learning rate', best_r)




#################train for 20 epochs######################################
def perceptrondev(index,value,data,length,r0):
	global count20
	random.seed(40)
	w=[0.005]*(length+1)
	for val in range(len(w)):
		w[val]=random.uniform(-0.01,0.01)
	a=[0.005]*(length+1)
	for val in range(len(a)):
		a[val]=random.uniform(-0.01,0.01)

	b=random.uniform(-0.01,0.01)
	ba=random.uniform(-0.01,0.01)
	product=copy.deepcopy(index)
	predicted=0
	bestweight=copy.deepcopy(index)
	bestbias=[]
	for item in range(len(product)):
		product[item]=product[item][:1]
		product[item][0]=0
	t=0
	for epoch in range(0,20):
		r=r0/1+t
		for item in range(len(index)):
			product[item][0]=0
			label=data[item][0]
			for i in range(len(index[item])):
				k=index[item][i]
				product[item][0]+=w[k]*value[item][i]
			predicted=product[item][0]+b
			if predicted >= 0:
				predictedlabel=1
			else:
				predictedlabel=0
			if predictedlabel==1 and label==0:
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=w[k]-r*(value[item][i])
				b=b-r
				count20[epoch]+=1
			elif predictedlabel==0 and label==1:
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=w[k]+r*(value[item][i])
				b=b+r
				count20[epoch]+=1
			else:
				for i in range(len(index[item])):
					k=index[item][i]
					w[k]=w[k]		
				b=b
			for val in range(len(a)):
				a[val]=a[val]+w[val]
			ba=ba+b			
		
		bestweight[epoch]=a
		bestbias.append(ba)
		c=list(zip(index,value,data))
		random.shuffle(c)
		index,value,data=list(zip(*c))
		t+=1
		
		
		
	return bestweight,bestbias




############ Testing  ###################
indextest,valuetest,datatest,lengthtest=extractfeatures(sys.argv[7])
indextrain, valuetrain, datatrain, lengthtrain=extractfeatures(sys.argv[6])
weighttrain,biastrain=perceptrondev(indextrain,valuetrain,datatrain,max(lengthtest,lengthtrain),best_r)

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




