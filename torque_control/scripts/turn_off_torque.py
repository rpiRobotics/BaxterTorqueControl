# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:07:36 2017

@author: cats
"""

import rospy
import baxter_interface

rospy.init_node('Hello_Baxter')
limb = baxter_interface.Limb('right')
