#!/usr/bin/env python
"""
Created on Wed Jan 18 13:28:20 2017

@author: Andrew Cunningham
@description: This script will append joint position/effort information to the 
"""

import rospy
import baxter_interface
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState
import time
from PID import PID

class PControlNode:
    def __init__(self, file_name = 'data.csv', max_torque = .5):
        self.max_torque = max_torque        
        
        rospy.init_node('Torque Recorder')

        ## FILE WRITING STUFF
        self.start_time = time.time()
        rospy.Subscriber('/robot/joint_states', JointState, self._joint_state_callback)
        self.f = open(file_name, 'a+')
        self.data = []
        self.new_data = False
        self.recorded_data = False
        
    def _joint_state_callback(self, data):
        if self.new_data:
            return
            
        self.time = time.time() - self.start_time
        self.data = []
        self.data.append(data.position[9:16])
        self.data.append(data.effort[9:16])
        self.data = [e for l in self.data for e in l]
        if len(self.data) > 5:
            self.new_data = True
    
    def _record_data(self):
        if not self.new_data:
            return
            
        print "RECORDING STUFF!"
        for datum in self.data:
            self.f.write(str(datum)+',')
        
        self.f.write('\n')    
        self.new_data = False
        self.recorded_data = True
    
    def spin(self):
        # RATE MUST BE AT LEAST 5 TO CANCEL GRAVITY COMP
        r = rospy.Rate(10)
        while not self.recorded_data:
            self._record_data()
            r.sleep()

my_node = PControlNode('/home/cats/cunnia3/nasa_ws/data/position1.csv')
my_node.spin()