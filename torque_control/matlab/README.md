## gravity_comp_identification.m
This script creates a gravity compensation model by fitting a model to data from pid_auto_big.csv.  

## baxter_gravity_comp.m
This script runs the compensated model on the actual Baxter robot.  
IMPORTANT: Requires a RR bridge that has gravity compensation cancellation or the arm will fly upwards! This happens because
it is effectively compensating for gravity twice.

## pid_auto_big.csv
A collection of data in the order of (s0, s1, e0, e1, w0, w1, w2) with 7 joint angles first and then 7 corresponding torques.
