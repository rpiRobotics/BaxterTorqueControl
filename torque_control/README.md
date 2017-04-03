## auto_recorder.py
This program collects data (angles and corresponding torques) on random configurations of the baxter arm.  If run, it will continue to move the Baxter arm into 
different random configurations (within joint limits) and then record the data to a CSV.  The data is recorded in the following
order (e0,e1,s0,s1,w0,w1,w2). Data recorded will include joint angles as the first 7 entries and torques as the last 7 entries.

## auto_recorder_pid.py
The program collects data (angles and corresponding torques) on random configurations of the baxter arm. In contrast to auto_recorder.py
it uses both a fit model and PD control to maintain the arm in the goal configuration.  Further, this script records the torque
commands sent rather than the torque values sensed. Data recorded will include joint angles as the first 7 entries and torques as the last 7 entries.

## nn_learn.py
Train a nn compensator for gravity compensation.  Takes as input
a full set of data, and a g-comp model from Matlab.  Saves a NN that compensates
for non-modeled error.  The path for loading data will need to be changed depending on where the data is located.  
NOTE: the data is expected to be in the following order (s0,s1,e0,e1,w0,w1,w2).  Make sure the appropriate columns are swapped
from the recording scripts.

## nn_compensation.py
Use a learnt NN compensator and a Matlab identified gravity compensation model to attempt to hold the arms in a stationary
position using torque control and no feedback.
