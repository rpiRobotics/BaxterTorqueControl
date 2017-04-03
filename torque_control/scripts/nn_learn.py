# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:37:28 2017

@author: Andrew Cunningham
@description: Train a nn compensator for gravity compensation.  Takes as input
a full set of data, and a g-comp model from Matlab.  Saves a NN that compensates
for non-modeled error  
"""

import sklearn
from sklearn.neural_network import MLPRegressor
import numpy as np
import math
import pickle
from sklearn.externals import joblib
import copy

def get_model_comp(q):
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    q5 = q[4]
    q6 = q[5]
    q7 = q[6]
    tau =  [0.0, q2*(-3.626267454553836e1)+math.cos(q2)*6.003144882868656+math.sin(q2)*4.501519840073571e1-math.cos(q2)*math.cos(q4)*1.331970459296284e1+math.cos(q3)*math.sin(q2)*4.674942117521753+math.cos(q2)*math.sin(q4)*6.955499209157561e-2-math.sin(q2)*math.sin(q3)*7.642332602263413e-1+math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))*1.143218232830278e-1-math.cos(q7)*(math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))+math.cos(q5)*math.sin(q2)*math.sin(q3))*2.968987099013326e-2+math.sin(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))*2.396559266157002-math.sin(q7)*(math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))+math.cos(q5)*math.sin(q2)*math.sin(q3))*4.293584044013006e-2+math.cos(q7)*(math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4)))*4.293584044013006e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4)))*2.968987099013326e-2+math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))*1.707237974575091e-1-math.cos(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4))*2.396559266157002-math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))*1.713781382792949e-1+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4))*1.143218232830278e-1-math.sin(q2)*math.sin(q3)*math.sin(q5)*1.707237974575091e-1+math.cos(q3)*math.cos(q4)*math.sin(q2)*6.955499209157561e-2+math.cos(q3)*math.sin(q2)*math.sin(q4)*1.331970459296284e1-math.cos(q5)*math.sin(q2)*math.sin(q3)*1.713781382792949e-1-5.039760974781792, math.cos(q2)*math.cos(q3)*7.642332602263413e-1+math.cos(q2)*math.sin(q3)*4.674942117521753+math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))*1.143218232830278e-1+math.cos(q7)*(math.cos(q2)*math.cos(q3)*math.cos(q5)-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5))*2.968987099013326e-2+math.sin(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))*2.396559266157002+math.sin(q7)*(math.cos(q2)*math.cos(q3)*math.cos(q5)-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5))*4.293584044013006e-2+math.cos(q7)*(math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6))*4.293584044013006e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6))*2.968987099013326e-2+math.cos(q2)*math.cos(q3)*math.cos(q5)*1.713781382792949e-1+math.cos(q2)*math.cos(q4)*math.sin(q3)*6.955499209157561e-2+math.cos(q2)*math.cos(q3)*math.sin(q5)*1.707237974575091e-1+math.cos(q2)*math.sin(q3)*math.sin(q4)*1.331970459296284e1+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3)*1.707237974575091e-1-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5)*1.713781382792949e-1+math.cos(q2)*math.cos(q6)*math.sin(q3)*math.sin(q4)*2.396559266157002-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6)*1.143218232830278e-1, q4*3.968736013465624e-2+math.cos(q4)*math.sin(q2)*6.955499209157561e-2+math.sin(q2)*math.sin(q4)*1.331970459296284e1+math.cos(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.707237974575091e-1+math.cos(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*2.396559266157002-math.sin(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.713781382792949e-1-math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.143218232830278e-1-math.cos(q7)*(math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.293584044013006e-2+math.sin(q7)*(math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.968987099013326e-2+math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.143218232830278e-1+math.cos(q5)*math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.396559266157002-math.cos(q7)*math.sin(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.968987099013326e-2-math.sin(q5)*math.sin(q7)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*4.293584044013006e-2-math.cos(q2)*math.cos(q3)*math.cos(q4)*1.331970459296284e1+math.cos(q2)*math.cos(q3)*math.sin(q4)*6.955499209157561e-2+1.852581172960766e-1, math.cos(q6)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*(-1.143218232830278e-1)-math.cos(q7)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*2.968987099013326e-2-math.sin(q6)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.396559266157002-math.sin(q7)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*4.293584044013006e-2-math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.713781382792949e-1-math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.707237974575091e-1+math.cos(q6)*math.sin(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.968987099013326e-2+math.cos(q2)*math.cos(q5)*math.sin(q3)*1.707237974575091e-1-math.cos(q2)*math.sin(q3)*math.sin(q5)*1.713781382792949e-1-math.cos(q6)*math.cos(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*4.293584044013006e-2, math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*2.396559266157002-math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*1.143218232830278e-1-math.cos(q7)*(math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))-math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.293584044013006e-2+math.sin(q7)*(math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))-math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.968987099013326e-2+math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.143218232830278e-1+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.396559266157002, math.cos(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*(-4.293584044013006e-2)+math.sin(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.968987099013326e-2-math.cos(q7)*(math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.968987099013326e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.293584044013006e-2]
    return tau        

def normalize_feature(xmin, xmax, x):
    """ Normalize x to bebetween -1 and 1"""
    a = -1
    b = 1
    return a + (x-xmin)*(b-a)/(xmax-xmin)

def scale_joint_angles(q):
    """ Scale Baxter joint angles"""
    q_scaled = copy.deepcopy(q)
    q_scaled[0] = normalize_feature(-1.7016 , 1.7016 , q[0])
    q_scaled[1] = normalize_feature(-2.147 , 1.047 , q[1])
    q_scaled[2] = normalize_feature(-3.0541  , 3.0541  , q[2])
    q_scaled[3] = normalize_feature( 	-0.05  , 2.618  , q[3])
    q_scaled[4] = normalize_feature( 	-3.059  ,  	3.059  , q[4])
    q_scaled[5] = normalize_feature(-1.5707  , 2.094  , q[5])
    q_scaled[6] = normalize_feature(-3.059 , 3.059, q[6])
    return q_scaled

# load data
full_data = np.genfromtxt('../../../data/pid_auto_big.csv', delimiter=',')

# subtact model from data
training_data = np.zeros_like(full_data)
for i in range(full_data.shape[0]):
    recorded_q = full_data[i,:7]
    recorded_tau = full_data[i,7:]
    difference = recorded_tau - get_model_comp(recorded_q)
    training_data[i, :7] = scale_joint_angles(recorded_q)
    training_data[i,7:] = np.asarray(difference)

# train NN
train_joint_angles = training_data[:,:7]
train_joint_torques = training_data[:,7:]

## TRAIN NN MODEL
my_regressor = MLPRegressor(hidden_layer_sizes=(7,7,7,7), activation='tanh', solver='lbfgs', \
max_iter=70000000, learning_rate = 'adaptive', verbose = True, alpha = 0.0005)
my_regressor.fit(train_joint_angles, train_joint_torques) 

## EVALUATE MODEL
error = np.zeros_like(train_joint_torques)
for i in range(full_data.shape[0]):
    recorded_q = scale_joint_angles(full_data[i,:7])
    compensated_tau = training_data[i,7:]
    error[i,:] = my_regressor.predict(recorded_q) - compensated_tau

# save NN
joblib.dump(my_regressor, 'nn_model.pkl') 