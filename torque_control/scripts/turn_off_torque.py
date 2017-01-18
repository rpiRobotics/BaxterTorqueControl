# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:07:36 2017

@author: Andrew Cunningham
@description: Turns off all torque to right arm. 
**CAUTION** ARM WILL DROP VIOLENTLY
"""
import rospy
import baxter_interface
from std_msgs.msg import Empty
import time

rospy.init_node('Hello_Baxter')
limb = baxter_interface.Limb('right')
torque = limb.joint_angles()

## SET TORQUES TO 0
torque['right_s0']=0.0
torque['right_s1']=0.0
torque['right_e0']=0.0
torque['right_e1']=0.0
torque['right_w0']=0.0
torque['right_w1']=0.0
torque['right_w2']=0.0

## DISABLE GRAVITY COMPENSATION
msg = Empty()
pub = rospy.Publisher('/robot/limb/right/suppress_gravity_compensation', Empty)

## WHERE THE ACTUAL WORK HAPPENS
now = time.time()
while time.time() - now < 20:
    limb.set_joint_torques(torque)
    pub.publish(msg)
    time.sleep(.1)