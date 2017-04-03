%%
% AUTHOR: Andrew Cunningham
% DESCRIPTION: Test of custom gravity compensation

clear
load('grav_comp')
g = matlabFunction(G_m);

robot = RobotRaconteur.Connect('tcp://192.168.1.127:44533/BaxterJointServer/Baxter');
% GET READY
pause(2);

% DO CUSTOM GRAVITY COMP
robot.setControlMode(uint8(2));

% COLLECT DATA
pid_joint_angles = robot.joint_positions
pid_joint_angles = pid_joint_angles(8:14)

pid_joint_torques = robot.joint_torques
pid_joint_torques = pid_joint_torques(8:14)


for i=1:1:10000
    q_all = robot.joint_positions;
    q_right = q_all(8:14);
    model_torque = g(q_right(2), q_right(3), q_right(4), q_right(5), q_right(6),  q_right(7));
    model_torque = model_torque
    model_angles = robot.joint_positions
    model_angles = model_angles(8:14)
    robot.stopGravityCompensation();
    
    robot.setJointCommand('right', model_torque);
    
    % SAFETY FIRST!
    if any(abs(model_torque) > 30) || any(abs(robot.joint_velocities)> 3.5)
        disp('STOPPING');
        robot.setControlMode(uint8(0));
        break
    end
    
    pause(.01)
end

% GO BACK TO START
robot.setControlMode(uint8(0));
[pid_joint_torques model_torque]
[pid_joint_angles model_angles]
robot.setJointCommand('right', zeros(7,1));
pause(5)