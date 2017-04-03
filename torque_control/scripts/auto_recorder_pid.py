#!/usr/bin/env python
"""
Created on Wed Jan 18 13:28:20 2017

@author: Andrew Cunningham
@description: This script will attempt to hold the joint positions at the start
of the script by doing proportional control. Run this script once for every position
that you'd like to include in the robot force calibration, it'll append the information
to the file each time
"""

import rospy
import baxter_interface
from std_msgs.msg import Empty
from sensor_msgs.msg import JointState
import time
from PID import PID
import numpy as np
import time
import math

class PControlNode:
    def __init__(self, file_name = 'data.csv', max_torque = 30):
        self.file_name = file_name
        self.max_torque = max_torque        
        
        rospy.init_node('Torque Recorder')
        self.limb = baxter_interface.Limb('right')
        self.desired_angles = self.limb.joint_angles()
        self.torques = self.limb.joint_angles()
        
        ## CONSTRUCT PID OBJECT FOR EACH JOINT
        self.pid_dict = {}               
        self.pid_dict['right_e0'] = PID(P=20.0, I=0.0, D=12.0, Derivator=0, Integrator=0, Integrator_max=0, Integrator_min=0)
        self.pid_dict['right_e1'] = PID(P=20.0, I=0.0, D=12.0, Derivator=0, Integrator=0, Integrator_max=0, Integrator_min=0)
        self.pid_dict['right_s0'] = PID(P=3.0, I=0.0, D=2.0, Derivator=0, Integrator=0, Integrator_max=0, Integrator_min=0)
        self.pid_dict['right_s1'] = PID(P=40.0, I=0.0, D=12.0, Derivator=0, Integrator=0, Integrator_max=0, Integrator_min=0)
        self.pid_dict['right_w0'] = PID(P=8.0, I=0.0, D=2.0, Derivator=0, Integrator=1, Integrator_max=4, Integrator_min=-4)
        self.pid_dict['right_w1'] = PID(P=5.0, I=0.0, D=2.0, Derivator=0, Integrator=1, Integrator_max=4, Integrator_min=-4)
        self.pid_dict['right_w2'] = PID(P=3.0, I=0.0, D=2.0, Derivator=0, Integrator=0, Integrator_max=0, Integrator_min=0)
        
        ## CANCEL GRAV COMP
        self.msg = Empty()
        self.pub = rospy.Publisher('/robot/limb/right/suppress_gravity_compensation', Empty)
        
        ## FILE WRITING STUFF
        self.start_time = time.time()
        rospy.Subscriber('/robot/joint_states', JointState, self._joint_state_callback)
        self.f = open(file_name, 'a+')
        self.data = []
        
        ## CONTROL VARIABLES
        self.arm_moving = True   # tells us if the arm is in motion (for recording)
        self.e_stop = False      # tells us if the arm is moving too fast
        self.end_round = False  # we have succseffuly recorded data
        self.num_safe_to_read = 0 # number of consecutive safe to read conditions
        self.get_grav_torques = True # Whether or not we should obtain new gravity torques        
        
        self.total_new_data = 0 # total of new pieces of information added        
        
        self.grav_comp_tau_dict = {}
        
        
    def _find_grav_comp_tau(self, arm='right'):
        """ Use Matlab's result to calculate the torques on joints due to gravity
        returns 7x1 torque list"""
        angles = self.limb.joint_angles()
        q2 = angles[arm+'_s1']
        q3 = angles[arm+'_e0']
        q4 = angles[arm+'_e1']
        q5 = angles[arm+'_w0']
        q6 = angles[arm+'_w1']
        q7 = angles[arm+'_w2']

        tau =  [0.0,q2*(-3.627472254963471e1)+math.cos(q2)*6.232481694270538+math.sin(q2)*4.489154234624326e1-math.cos(q2)*math.cos(q4)*1.318641850084027e1+math.cos(q3)*math.sin(q2)*4.727636627170416+math.cos(q2)*math.sin(q4)*2.285886669516682e-1-math.sin(q2)*math.sin(q3)*7.702455065557092e-1+math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))*8.89930982816937e-2-math.cos(q7)*(math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))+math.cos(q5)*math.sin(q2)*math.sin(q3))*2.882669690093901e-2+math.sin(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))*2.400275531283779-math.sin(q7)*(math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))+math.cos(q5)*math.sin(q2)*math.sin(q3))*4.470495569536943e-2+math.cos(q7)*(math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4)))*4.470495569536943e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))-math.sin(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4)))*2.882669690093901e-2+math.cos(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))*1.95529724120771e-1-math.cos(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4))*2.400275531283779-math.sin(q5)*(math.cos(q2)*math.sin(q4)+math.cos(q3)*math.cos(q4)*math.sin(q2))*1.647424686660705e-1+math.sin(q6)*(math.cos(q2)*math.cos(q4)-math.cos(q3)*math.sin(q2)*math.sin(q4))*8.89930982816937e-2-math.sin(q2)*math.sin(q3)*math.sin(q5)*1.95529724120771e-1+math.cos(q3)*math.cos(q4)*math.sin(q2)*2.285886669516682e-1+math.cos(q3)*math.sin(q2)*math.sin(q4)*1.318641850084027e1-math.cos(q5)*math.sin(q2)*math.sin(q3)*1.647424686660705e-1-5.190181465211625,math.cos(q2)*math.cos(q3)*7.702455065557092e-1+math.cos(q2)*math.sin(q3)*4.727636627170416+math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))*8.89930982816937e-2+math.cos(q7)*(math.cos(q2)*math.cos(q3)*math.cos(q5)-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5))*2.882669690093901e-2+math.sin(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))*2.400275531283779+math.sin(q7)*(math.cos(q2)*math.cos(q3)*math.cos(q5)-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5))*4.470495569536943e-2+math.cos(q7)*(math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6))*4.470495569536943e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q2)*math.cos(q3)*math.sin(q5)+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3))-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6))*2.882669690093901e-2+math.cos(q2)*math.cos(q3)*math.cos(q5)*1.647424686660705e-1+math.cos(q2)*math.cos(q4)*math.sin(q3)*2.285886669516682e-1+math.cos(q2)*math.cos(q3)*math.sin(q5)*1.95529724120771e-1+math.cos(q2)*math.sin(q3)*math.sin(q4)*1.318641850084027e1+math.cos(q2)*math.cos(q4)*math.cos(q5)*math.sin(q3)*1.95529724120771e-1-math.cos(q2)*math.cos(q4)*math.sin(q3)*math.sin(q5)*1.647424686660705e-1+math.cos(q2)*math.cos(q6)*math.sin(q3)*math.sin(q4)*2.400275531283779-math.cos(q2)*math.sin(q3)*math.sin(q4)*math.sin(q6)*8.89930982816937e-2,math.cos(q4)*math.sin(q2)*2.285886669516682e-1+math.sin(q2)*math.sin(q4)*1.318641850084027e1+math.cos(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.95529724120771e-1+math.cos(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*2.400275531283779-math.sin(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*1.647424686660705e-1-math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*8.89930982816937e-2-math.cos(q7)*(math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.470495569536943e-2+math.sin(q7)*(math.sin(q6)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.882669690093901e-2+math.cos(q5)*math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*8.89930982816937e-2+math.cos(q5)*math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.400275531283779-math.cos(q7)*math.sin(q5)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.882669690093901e-2-math.sin(q5)*math.sin(q7)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*4.470495569536943e-2-math.cos(q2)*math.cos(q3)*math.cos(q4)*1.318641850084027e1+math.cos(q2)*math.cos(q3)*math.sin(q4)*2.285886669516682e-1,math.cos(q6)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*(-8.89930982816937e-2)-math.cos(q7)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*2.882669690093901e-2-math.sin(q6)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.400275531283779-math.sin(q7)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*4.470495569536943e-2-math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.647424686660705e-1-math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))*1.95529724120771e-1+math.cos(q6)*math.sin(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.882669690093901e-2+math.cos(q2)*math.cos(q5)*math.sin(q3)*1.95529724120771e-1-math.cos(q2)*math.sin(q3)*math.sin(q5)*1.647424686660705e-1-math.cos(q6)*math.cos(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*4.470495569536943e-2,math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*2.400275531283779-math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))*8.89930982816937e-2-math.cos(q7)*(math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))-math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.470495569536943e-2+math.sin(q7)*(math.sin(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))-math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.882669690093901e-2+math.cos(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*8.89930982816937e-2+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4))*2.400275531283779,math.cos(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*(-4.470495569536943e-2)+math.sin(q7)*(math.sin(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))-math.cos(q2)*math.cos(q5)*math.sin(q3))*2.882669690093901e-2-math.cos(q7)*(math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*2.882669690093901e-2-math.sin(q7)*(math.cos(q6)*(math.cos(q5)*(math.sin(q2)*math.sin(q4)-math.cos(q2)*math.cos(q3)*math.cos(q4))+math.cos(q2)*math.sin(q3)*math.sin(q5))+math.sin(q6)*(math.cos(q4)*math.sin(q2)+math.cos(q2)*math.cos(q3)*math.sin(q4)))*4.470495569536943e-2]
        
        self.grav_comp_tau_dict[arm+'_s0']=0
        self.grav_comp_tau_dict[arm+'_s1']=tau[1]
        self.grav_comp_tau_dict[arm+'_e0']=tau[2]
        self.grav_comp_tau_dict[arm+'_e1']=tau[3]
        self.grav_comp_tau_dict[arm+'_w0']=tau[4]
        self.grav_comp_tau_dict[arm+'_w1']=tau[5]
        self.grav_comp_tau_dict[arm+'_w2']=tau[6]
     
    def _joint_state_callback(self, data):
        """ Handle safety checks and determine when to write information"""
        ## CHECK TO SEE IF WE SHOULD STOP
        if len(data.effort) < 10:
            return
   
        one_joint_too_fast = False
        velocities = data.velocity[9:16]
        for v in velocities:
            if abs(v) > 1.5:
                print "STOP!!!!!!!"
                self.e_stop = True
                
            if abs(v) > .05:
                one_joint_too_fast = True 
                self.arm_moving = True
                
        # If none of our joints voilate the threshold, the arm isnt moving
        if not one_joint_too_fast:
            self.arm_moving = False
    
    def _cancel_grav_comp(self):
        if not self.e_stop:
            self.pub.publish(self.msg)

    def _set_pid_to_current(self):
        """ set PID set points to current angles """
        current_angles = self.limb.joint_angles()
        for key in current_angles:
            self.pid_dict[key].setPoint(current_angles[key])

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
            elif 's0' in key:
                candidate[key] =np.random.uniform(0 , 0.1)
                
        return candidate
        
    def _move_to_random(self):
        """ Attempt to move to random configuration, but only record data if close enough"""
        desired_angles = self._generate_random_joint_config()
            
        ## TRY TO MOVE TO POSITION IN AT MOST 5 SECONDS
        now = time.time()
        total_difference = 999
        # MOVE 
        while time.time() - now < 4 and total_difference > .1:
            self.limb.set_joint_positions(desired_angles)
            current_angles = self.limb.joint_angles()
            
            total_difference = 0
            for key in desired_angles.keys():
                total_difference += abs(current_angles[key] - desired_angles[key])
            
            ## MAKE SURE ARM IS IN SAFE POSITION
            pose = self.limb.endpoint_pose() 
            dist_from_wheel = math.sqrt((pose['position'].x - .2)**2  + (pose['position'].y - -.28)**2    + (pose['position'].z - 0.413)**2)
            if pose['position'].z < -.3 or dist_from_wheel < .3:
                # go back to start
                angles = self.limb.joint_angles()
                angles = dict.fromkeys(angles, 0)
                self.limb.move_to_joint_positions(angles, threshold = .04)
                self.end_round = True
                return
            
        self.end_round = False
        time.sleep(1)
        
    def pcontrol_torques(self):
        """ Read joint position and then apply torque according to P control """
        angles = self.limb.joint_angles()
        
        # Safety measure, if the arm is moving to fast, stop torque control
        if self.e_stop:
            self.limb.move_to_joint_positions(self.desired_angles)
            self.end_program = True
            return
        
        ## GET ESTIMATED GRAV COMP        
        self._find_grav_comp_tau()
        
        for key in angles:
            torque = self.pid_dict[key].update(angles[key]) + self.grav_comp_tau_dict[key]
            if torque > self.max_torque:
                torque = self.max_torque
            elif torque < -self.max_torque:
                torque = -self.max_torque  
            self.torques[key] = torque
            print key, "%.2f" % self.pid_dict[key].error, torque
        
        #print "\n"
        self.limb.set_joint_torques(self.torques)
        return
        
    def attempt_record(self):
        """ Only record data if the arm is not moving and enough time has passed """
        if time.time() - self.start_time > 2 and not self.arm_moving:
            self.num_safe_to_read+=1
            #print self.num_safe_to_read
            if self.num_safe_to_read > 30:
                angles = self.limb.joint_angles()
                data = []            
                
                data.append(angles['right_e0'])
                data.append(angles['right_e1'])
                data.append(angles['right_s0'])
                data.append(angles['right_s1'])
                data.append(angles['right_w0'])
                data.append(angles['right_w1'])
                data.append(angles['right_w2'])
                
                data.append(self.torques['right_e0'])
                data.append(self.torques['right_e1'])
                data.append(self.torques['right_s0'])
                data.append(self.torques['right_s1'])
                data.append(self.torques['right_w0'])
                data.append(self.torques['right_w1'])
                data.append(self.torques['right_w2'])
                
                
                self.f = open(self.file_name, 'a+')
                for datum in data:
                    self.f.write(str(datum)+',')
            
                self.f.write('0\n')  
                self.f.close()
                self.end_round = True
                print "WROTE DATA!"
                self.total_new_data += 1
                self.num_safe_to_read = 0
        else:
            self.num_safe_to_read = 0
    
    def spin(self):
        # RATE MUST BE AT LEAST 5 TO CANCEL GRAVITY COMP
        r = rospy.Rate(50)
        while not rospy.is_shutdown() and self.total_new_data < 100:
            self._move_to_random()
            self._set_pid_to_current()
            self.end_round = False
            self.e_stop = False
            self.get_grav_torques = True
            r.sleep()
            r.sleep()
            while not self.e_stop and not self.end_round:
                r.sleep()
                self._cancel_grav_comp() 
                self.pcontrol_torques()
                self.attempt_record()
                
        self.f.close()

my_node = PControlNode('/home/cats/cunnia3/nasa_ws/data/pid_auto.csv')
my_node.spin()
