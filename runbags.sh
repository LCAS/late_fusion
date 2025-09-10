#!/bin/bash

source /home/user/ros2_ws/install/setup.bash
cd /home/user/bags

while true; do
  for rosbag in *.db3/; do 
    ros2 bag play $rosbag --clock
  done
done
