import random


#-------------------------------------
# Split the train data LIBSVM file 
# into 5 CV Splits
#-------------------------------------

def gy(filename,file2,file3,file4,file5,file6):
	f=open(filename,'r')
	data=f.read().strip()
	data=data.split('\n')
	m=len(data)
	data0=data[:m/5]
	data1=data[m/5:2*m/5]
	data2=data[2*m/5:3*m/5]
	data3=data[3*m/5:4*m/5]
	data4=data[4*m/5:]
	

	fi=open(file2,'w')
	for line in data:
		fi.write(line)
		fi.write('\n')
	fi1=open(file3,'w')
	for line in data1:
		fi1.write(line)
		fi1.write('\n')
	fi2=open(file4,'w')
	for line in data2:
		fi2.write(line)
		fi2.write('\n')
	fi3=open(file5,'w')
	for line in data3:
		fi3.write(line)
		fi3.write('\n')
	fi4=open(file6,'w')
	for line in data4:
		fi4.write(line)
		fi4.write('\n')

gy('data/extracted_features_train.csv','data/split1.csv','data/split2.csv','data/split3.csv','data/split4.csv','data/split5.csv')