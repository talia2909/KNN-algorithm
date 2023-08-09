import json
import numpy as np   # check out how to install numpy
from utils import load, plot_sample

# =========================================
#       Homework on K-Nearest Neighbors
# =========================================
# Course: Introduction to Information Theory
# Lecturer: Haim H. Permuter.
#
# NOTE:
# -----
# Please change the variable ID below to your ID number as a string.
# Please do it now and save this file before doing the assignment

ID = '316471978'

# Loading and plot a sample of the data
# ---------------------------------------
# The MNIST database contains in total of 60000 train images and 10000 test images
# of handwritten decision.
# This database is a benchmark for many classification algorithms, including neural networks.
# For further reading, visit http://yann.lecun.com/exdb/mnist/
#
# You will implement the KNN algorithm to classify two decision from the database: 3 and 5.
# First, we will load the data from .MAT file we have prepared for this assignment: MNIST_3_and_5.mat

Xtrain, Ytrain, Xvalid, Yvalid, Xtest = load('MNIST_3_and_5.mat')

# The data is divided into 2 pairs:
# (Xtrain, Ytrain) , (Xvalid, Yvalid)
# In addition, you have unlabeled test sample in Xtest.
#
# Each row of a X matrix is a sample (gray-scale picture) of dimension 784 = 28^2,
# and the digit (number) is in the corresponding row of the Y vector.
#
# To plot the digit, see attached function 'plot_sample.py'

sampleNum = 0
plot_sample(Xvalid[sampleNum, :], Yvalid[sampleNum, :])

# Build a KNN classifier based on what you've learned in class:
#
# 1. The classifier should be given a train dataset (Xtrain, Ytain),  and a test sample Xtest.
# 2. The classifier should return a class for each row in test sample in a column vector Ytest.
#
# Finally, your results will be saved into a <ID>.txt file, where each row is an element in Ytest.
#
# Note:
# ------
# For you conveniece (and ours), give you classifications in a 1902 value 1 vector named Ytest,
# and set the variable ID at the beginning of this file to your ID.


# < your code here >
lenX_v=len(Xvalid[:, 1])
lenX_t=len(Xtrain[:,1])
dis_Xt_Xv=np.array([]) #the distance between the twp sumples : Xtrial and Xvalid.
Vec_num=np.array([]) #the array of the the Ytrial values that mach each distance.
decision=np.array([]) #the array of the decisions for each sample.
idx=np.array([]) #an index array of the distances value
all_k=[]
counter=0 #counter of success for each k
for k in range(30) :
    for i in range(lenX_v):
        for j in range(lenX_v):
            dis_Xt_Xv=np.append(dis_Xt_Xv, [np.linalg.norm(Xtrain[j, :] - Xvalid[i, :])]) # calaulate the norm of a vector using the Euclidean formula.
        idx=np.argpartition(dis_Xt_Xv, k) # sorting the k lowest neighbors at the stats of the array.
        idx=idx[:k+1] #an index array of the k nearest neighbors.
        for value in idx:
            Vec_num=np.append(Vec_num, [Ytrain[value]]) #adding value for each distance according to Ytrain.
        Vec_num=Vec_num.astype(int)
        decision=np.append(decision,np.bincount(Vec_num).argmax())
        dis_Xt_Xv=np.array([])
        Vec_num=np.array([])
    for i in range(lenX_v):
        if decision[i]==Yvalid[i]: #success
            counter+=1 #counting how many success have been for this k.
    all_k=np.append(all_k, counter)
    counter=0
    decision=np.array([])
print(all_k)
k=15
#real tast :
lenX_test=len(Xtest[:,1])
dis_Xtest_Xtrial=np.array([]) #the distance between the twp sumples : Xtrial and Xvalid.
Vec_num=np.array([]) #the array of the the Ytrial values that mach each distance.
Ytest=np.array([]) #the array of the decisions for each sample.
idx=np.array([]) #an index array of the distances value
for i in range(lenX_test):
    for j in range(lenX_test):
        dis_Xtest_Xtrial=np.append(dis_Xtest_Xtrial, [np.linalg.norm(Xtrain[j, :] - Xtest[i, :])]) # calaulate the norm of a vector using the Euclidean formula.
    idx=np.argpartition(dis_Xtest_Xtrial, k) # sorting the k lowest neighbors at the stats of the array.
    idx=idx[:k+1] #an index array of the k nearest neighbors.
    for value in idx:
        Vec_num=np.append(Vec_num, [Ytrain[value]]) #adding value for each distance according to Ytrain.
    Vec_num=Vec_num.astype(int)
    Ytest=np.append(Ytest, [np.bincount(Vec_num).argmax()])
    dis_Xtest_Xtrial=np.array([])
    Vec_num=np.array([])
# save classification results
print('saving')
np.savetxt(f'{ID}.txt', Ytest, delimiter=", ", fmt='%i')
print('done')
