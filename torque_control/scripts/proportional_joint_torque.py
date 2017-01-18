# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:28:20 2017

@author: Andrew Cunningham
@description: This script will attempt to hold the joint positions at the start
of the script by doing proportional control. Start with proportional control of 
w1
"""

import rospy
import baxter_interface
from std_msgs.msg import Empty
import time
from PID import PID

class PControlNode:
    def __init__(self, max_torque = 4):
        self.max_torque = max_torque        
        
        rospy.init_node('Hello_Baxter')
        self.limb = baxter_interface.Limb('right')
        self.desired_angles = self.limb.joint_angles()
        self.desired_angles['right_s0']=0.0
        self.desired_angles['right_s1']=0.0
        self.desired_angles['right_e0']=0.0
        self.desired_angles['right_e1']=0.0
        self.desired_angles['right_w0']=0.0
        self.desired_angles['right_w1']=0.0
        self.desired_angles['right_w2']=0.0
        
        self.torques = self.limb.joint_angles()
        
        ## CONSTRUCT PID OBJECT FOR EACH JOINT
        self.pid_dict = {}        
        for key in self.desired_angles:
            self.pid_dict[key] = PID(P=5.0, I=0.0, D=4.0, Derivator=0, Integrator=0, Integrator_max=0, Integrator_min=0)
            self.pid_dict[key].setPoint(self.desired_angles[key])
        
        ## CANCEL GRAV COMP
        self.msg = Empty()
        self.pub = rospy.Publisher('/robot/limb/right/suppress_gravity_compensation', Empty)
        
    def _cancel_grav_comp(self):
        self.pub.publish(self.msg)
        
    def update(self):
        """ Read joint position and then apply torque according to P control """
        angles = self.limb.joint_angles()
        
        for key in angles:
            torque = self.pid_dict[key].update(angles[key])  
            print torque
            if torque > self.max_torque:
                torque = self.max_torque
            elif torque < -self.max_torque:
                torque = -self.max_torque  
            self.torques[key] = torque
                
        self.limb.set_joint_torques(self.torques)
        return
    
    def spin(self):
        # RATE MUST BE AT LEAST 5 TO CANCEL GRAVITY COMP
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self._cancel_grav_comp
            self.update()
            r.sleep()

my_node = PControlNode()
my_node.spin()