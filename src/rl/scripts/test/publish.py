#!/usr/bin/env python3

import rospy
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

class JointTrajectoryActionClient:
    def __init__(self):
        rospy.init_node('joint_trajectory_action_client', anonymous=True)
        self.pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=10)

        self.traj = JointTrajectory()
        self.traj.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        self.point = JointTrajectoryPoint()
        self.point.positions = [0.0] * 6
        self.point.velocities = [0.0] * 6
        self.point.accelerations = [0.0] * 6
        self.point.effort = [0.0] * 6
        self.point.time_from_start = rospy.Duration(3.0)
        
    def send_goal(self, positions, time_sec):
        self.point.positions = positions
        self.point.time_from_start = rospy.Duration(time_sec)

        self.traj.points.append(self.point)

        self.pub.publish(self.traj)
        rospy.loginfo("JointTrajectory message published.")

def main():
    traj_client = JointTrajectoryActionClient()
    traj_client.send_goal([0.0, -1.57, 1.57, 0.0, 0.0, 0.0], 3.0)

if __name__ == "__main__":
    while rospy.ROSInterruptException:
        main()