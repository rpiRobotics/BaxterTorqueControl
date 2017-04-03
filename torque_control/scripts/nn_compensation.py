# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 13:17:09 2017

@author: cats
@description: Use model based grav comp in conjunction with NN compensator
"""
import sklearn
from sklearn.neural_network import MLPRegressor
import numpy as np
import rospy
import baxter_interface
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState
import math
from sklearn.externals import joblib
import time
import copy

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
    
class GravCompModel():
    """ Use a Matlab-fit model and NN compensator to preform gravity compensation """
    def __init__(self, file_name='../../../data/auto_data.csv'):
        
        ## TRAIN NN MODEL
        self.my_nn_regressor = joblib.load(file_name)
        
            
    def get_model_comp(self, q):
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]
        q5 = q[4]
        q6 = q[5]
        q7 = q[6]
        tau =  [0.0, q2*(-3.626267454553836e1)+math.cos(q2)*6.003144882868656+math.sin(q2)*4.501519840073571e1-math.cos(q2)*math.cos(q4)*1.331970459296284e1+math.cos(q3)*math.sin(q2)*4.674942117521753+math.cos(q2)*math.sin(q4)*6.955499209157561e-2-math.sin(q2)*math.sin(q3)*7.642332602263413e-1+math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))*1.143218232830278e-1-math.cos(q7)*(math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))+math.cos(q5)*math.sin(q2)*math.sin(q3))*2.968987099013326e-2+math.sin(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))*2.396559266157002-math.sin(q7)*(math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))+math.cos(q5)*math.sin(q2)*math.sin(q3))*4.293584044013006e-2+math.cos(q7)*(math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4)))*4.293584044013006e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4)))*2.968987099013326e-2+math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))*1.707237974575091e-1-math.cos(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4))*2.396559266157002-math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))*1.713781382792949e-1+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4))*1.143218232830278e-1-math.sin(q2)*math.sin(q3)*math.sin(q5)*1.707237974575091e-1+math.cos(q3)*math.cos(q4)*math.sin(q2)*6.955499209157561e-2+math.cos(q3)*math.sin(q2)*math.sin(q4)*1.331970459296284e1-math.cos(q5)*math.sin(q2)*math.sin(q3)*1.713781382792949e-1-5.039760974781792, math.cos(q2)*math.cos(q3)*7.642332602263413e-1+math.cos(q2)*math.sin(q3)*4.674942117521753+math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))*1.143218232830278e-1+math.cos(q7)*(math.cos(q2)*math.cos(q3)*math.cos(q5)-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5))*2.968987099013326e-2+math.sin(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))*2.396559266157002+math.sin(q7)*(math.cos(q2)*math.cos(q3)*math.cos(q5)-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5))*4.293584044013006e-2+math.cos(q7)*(math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6))*4.293584044013006e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6))*2.968987099013326e-2+math.cos(q2)*math.cos(q3)*math.cos(q5)*1.713781382792949e-1+math.cos(q2)*math.cos(q4)*math.sin(q3)*6.955499209157561e-2+math.cos(q2)*math.cos(q3)*math.sin(q5)*1.707237974575091e-1+math.cos(q2)*math.sin(q3)*math.sin(q4)*1.331970459296284e1+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3)*1.707237974575091e-1-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5)*1.713781382792949e-1+math.cos(q2)*math.cos(q6)*math.sin(q3)*math.sin(q4)*2.396559266157002-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6)*1.143218232830278e-1, q4*3.968736013465624e-2+math.cos(q4)*math.sin(q2)*6.955499209157561e-2+math.sin(q2)*math.sin(q4)*1.331970459296284e1+math.cos(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.707237974575091e-1+math.cos(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*2.396559266157002-math.sin(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.713781382792949e-1-math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.143218232830278e-1-math.cos(q7)*(math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.293584044013006e-2+math.sin(q7)*(math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.968987099013326e-2+math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.143218232830278e-1+math.cos(q5)*math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.396559266157002-math.cos(q7)*math.sin(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.968987099013326e-2-math.sin(q5)*math.sin(q7)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*4.293584044013006e-2-math.cos(q2)*math.cos(q3)*math.cos(q4)*1.331970459296284e1+math.cos(q2)*math.cos(q3)*math.sin(q4)*6.955499209157561e-2+1.852581172960766e-1, math.cos(q6)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*(-1.143218232830278e-1)-math.cos(q7)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*2.968987099013326e-2-math.sin(q6)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.396559266157002-math.sin(q7)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*4.293584044013006e-2-math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.713781382792949e-1-math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.707237974575091e-1+math.cos(q6)*math.sin(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.968987099013326e-2+math.cos(q2)*math.cos(q5)*math.sin(q3)*1.707237974575091e-1-math.cos(q2)*math.sin(q3)*math.sin(q5)*1.713781382792949e-1-math.cos(q6)*math.cos(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*4.293584044013006e-2, math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*2.396559266157002-math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*1.143218232830278e-1-math.cos(q7)*(math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))-math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.293584044013006e-2+math.sin(q7)*(math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))-math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.968987099013326e-2+math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.143218232830278e-1+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.396559266157002, math.cos(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*(-4.293584044013006e-2)+math.sin(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.968987099013326e-2-math.cos(q7)*(math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.968987099013326e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.293584044013006e-2]
        return tau      
    
    def compute_torques(self, angle_dict_in):
        """ Use Matlab and then NN model to preform grav comp 
        @angles: dictionary relating joint name to angle"""
        processed_torques = {}
        angles = [0]*7
        
        angles[0] = angle_dict_in['right_s0']
        angles[1] = angle_dict_in['right_s1']
        angles[2] = angle_dict_in['right_e0']
        angles[3] = angle_dict_in['right_e1']
        angles[4] = angle_dict_in['right_w0']
        angles[5] = angle_dict_in['right_w1']
        angles[6] = angle_dict_in['right_w2']

        model_torques = self.get_model_comp(np.asarray((angles)))
        angles[0] = 0
        nn_torques = self.my_nn_regressor.predict(scale_joint_angles(angles))

        processed_torques['right_s0'] = model_torques[0]# + nn_torques[0,0]
        processed_torques['right_s1'] = model_torques[1] + nn_torques[0,1]
        processed_torques['right_e0'] = model_torques[2] + nn_torques[0,2]
        processed_torques['right_e1'] = model_torques[3] + nn_torques[0,3]
        processed_torques['right_w0'] = model_torques[4] + nn_torques[0,4]
        processed_torques['right_w1'] = model_torques[5]# + nn_torques[0,5]
        processed_torques['right_w2'] = model_torques[6]# + nn_torques[0,6] 
        return processed_torques
            
class GravityCompensatorNode:   
    def __init__(self, g_comp_model):
        rospy.init_node('svm_g_comp')
        self.g_comp_model = g_comp_model
        self.limb = baxter_interface.Limb('right')
        self.current_angles = []
        rospy.Subscriber('/robot/joint_states', JointState, self._joint_state_callback)
        self.pub = rospy.Publisher('/robot/limb/right/suppress_gravity_compensation', Empty)
               
        self.desired_angles = self.limb.joint_angles()
        
        ## CONTROL VARIABLES
        self.e_stop = False
        return
        
    def _joint_state_callback(self, data):
        """ Handle joint state info coming in, safety check """                                 
        # SAFETY MEASURE
        velocities = data.velocity[9:16]
        for v in velocities:
            if abs(v) > 3.5:
                self.e_stop = True
                        
    def _cancel_grav_comp(self):
        msg = Empty()
        if not self.e_stop:
            self.pub.publish(msg)        
        
    def update_torques(self):
        """ Read in new angles and send them to the compensator to publish new torques"""
        # Safety measure, if the arm is moving to fast, stop torque control
        if self.e_stop:
            print "STOP!!!!!!!"
            self.limb.move_to_joint_positions(self.desired_angles)
            return
            
        self.current_angles = self.limb.joint_angles()
        new_torques = self.g_comp_model.compute_torques(self.current_angles)
        print new_torques
        for key in new_torques.keys():
            print key,  new_torques[key]
            
        self.limb.set_joint_torques(new_torques)
        
    def spin(self):
        r = rospy.Rate(20)
        start = time.time()
        while not rospy.is_shutdown() and time.time() - start < 20:
            r.sleep()
            self._cancel_grav_comp()
            self.update_torques()

my_g_comp = GravCompModel('nn_model.pkl')
my_node = GravityCompensatorNode(my_g_comp)
my_node.spin()
