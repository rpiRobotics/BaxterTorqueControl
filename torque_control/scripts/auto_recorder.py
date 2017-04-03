#!/usr/bin/env python
"""
Created on Wed Jan 18 13:28:20 2017

@author: Andrew Cunningham
@description: This script will append joint position/effort information to the 
file by going to random joint configurations
"""

import rospy
import baxter_interface
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState
import numpy as np
import time
import math
from PID import PID

class PControlNode:
    def __init__(self, file_name = 'data.csv', max_torque = .5, limb_name = 'right'):
        self.max_torque = max_torque        
        
        rospy.init_node('torque_recorder')

        ## FILE WRITING STUFF
        self.start_time = time.time()
        rospy.Subscriber('/robot/joint_states', JointState, self._joint_state_callback)
        self.f = open(file_name, 'a+')
        self.data = []
        self.new_data = False       # prevent overflow, we get LOTS of joint information
        
        ## RANDOM MOVEMENT STUFF
        self.safe_to_record = False # ONLY RECORD WHEN WE ARE IN PLACE
        self.limb = baxter_interface.Limb('right')

        
    def _joint_state_callback(self, data):
        """ Handle joint state info coming in """
        if not self.safe_to_record:
            return
            
        if len(data.effort) > 8:
            self.new_data = True
        else:
            return            
            
        self.data = []
        self.data.append(data.position[9:16])
        self.data.append(data.effort[9:16])
        self.data = [e for l in self.data for e in l]
        
    def _generate_random_joint_config(self):
        """ Generate safe set of angles to go to """
        candidate=self.limb.joint_angles()
        candidate['right_s0'] = 0
        
        ## ASSIGN EACH JOINT A RANDOM ANGLE WITHIN BOUNDS
        for key in candidate.keys():
            if 's1' in key:                
                candidate[key] =np.random.uniform(-2.147 , 1.047)
            elif 'e0' in key:
                candidate[key] =np.random.uniform(-3.0541  , 3.0541)
            elif 'e1' in key:
                candidate[key] =np.random.uniform(-0.05  , 2.618)
            elif 'w0' in key:
                candidate[key] =np.random.uniform(-3.059 , 3.059 )
            elif 'w1' in key:
                candidate[key] =np.random.uniform(-1.5707  , 2.094)
            elif 'w2' in key:
                candidate[key] =np.random.uniform(-3.059 , 3.059)

        return candidate
    
    def _move_to_random(self):
        """ Attempt to move to random configuration, but only record data if close enough"""
        desired_angles = self._generate_random_joint_config()
            
        ## TRY TO MOVE TO POSITION IN AT MOST 5 SECONDS
        now = time.time()
        total_difference = 999
        # MVOE
        while time.time() - now < 4 and total_difference > .1:
            self.limb.set_joint_positions(desired_angles)
            current_angles = self.limb.joint_angles()
            
            total_difference = 0
            for key in desired_angles.keys():
                total_difference += abs(current_angles[key] - desired_angles[key])
            
            ## MAKE SURE ARM IS IN SAFE POSITION
            pose = self.limb.endpoint_pose() 
            dist_from_wheel = math.sqrt((pose['position'].x - .2)**2  + (pose['position'].y - -.28)**2    + (pose['position'].z - 0.413)**2)
            if pose['position'].z < -.3 or dist_from_wheel < .4:
                # go back to start
                angles = self.limb.joint_angles()
                angles = dict.fromkeys(angles, 0)
                self.limb.move_to_joint_positions(angles, threshold = .04)
                return
        
        self.safe_to_record = True
        time.sleep(2)
                
    def _record_data(self):
        self._move_to_random()
        
        if not self.safe_to_record:
            return
        
        print "RECORDING STUFF!"
        for datum in self.data:
            self.f.write(str(datum)+',')
        
        self.f.write('0\n')    
        self.f.flush()
        self.new_data = False
        self.safe_to_record = False
    
    def spin(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self._record_data()
            r.sleep()

my_node = PControlNode('/home/cats/cunnia3/nasa_ws/data/auto_data_2_27.csv')
my_node.spin()