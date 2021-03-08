#!/usr/bin/env python
# rospy for the subscriber and publisher
import rospy
# type of messages we are going to use:
# this is the command to the motors:
from std_msgs.msg import Float64
# this is the command of velocity from the keyboard node:
from geometry_msgs.msg import Twist
# Miscellaneous libraries that we might use:
import matplotlib.pyplot as plt
import numpy as np

left_wheel = 0
right_wheel = 0

def velocity_callback(msg):
    tw = 0.2 #trackwidth of the rover
    x_dot = msg.linear.x
    psi_dot = msg.angular.z

    #Motors' velocities:
    global left_wheel, right_wheel
    r = 0.03 #radius of wheels
    left_wheel = x_dot - psi_dot*tw/2
    left_wheel = left_wheel/r
    right_wheel = x_dot + psi_dot*tw/2
    right_wheel = right_wheel/r
    

def main():
    global left_wheel, right_wheel
    rospy.init_node('VelocitiesConverter')
    rospy.Subscriber("/cmd_vel", Twist, velocity_callback)
    RW_cmd_topic = "/rover/joint_right_wheel_velocity_controller/command"
    LW_cmd_topic = "/rover/joint_left_wheel_velocity_controller/command"
    RW_pub = rospy.Publisher(RW_cmd_topic, Float64, queue_size = 1)
    LW_pub = rospy.Publisher(LW_cmd_topic, Float64, queue_size = 1)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
	RW_pub.publish(right_wheel)
        LW_pub.publish(left_wheel)
	print("Converting keyboard commands to motor commands")
        rate.sleep()

if __name__ == '__main__':
    main()
