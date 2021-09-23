#!/bin/zsh -e
gnome-terminal -e 'roscore'
sleep 3
#cd /home/wang/Hunter_ws ;exec zsh
#source ./devel/setup.zsh
gnome-terminal --working-directory=$PWD -e '/bin/bash -c "source devel/setup.bash;roslaunch rvo_ros rvo_gazebo_agent.launch"'
sleep 5
gnome-terminal --working-directory=$PWD -e '/bin/bash -c "source devel/setup.bash;rosrun rvo_ros set_goals_client random -7.5 7.5 -7.5 7.5"'
sleep 2
gnome-terminal --working-directory=$PWD -e '/bin/bash -c "source devel/setup.bash;roslaunch husky_viz view_robot.launch"'
sleep 2
gnome-terminal --working-directory=$PWD -e '/bin/bash -c "source devel/setup.bash;roslaunch husky_navigation amcl_demo.launch"'
sleep 2
gnome-terminal --working-directory=$PWD -e '/bin/bash -c "source devel/setup.bash;rosrun hunter_listener_node hunter_tf_listener.py"'
sleep 2
gnome-terminal --working-directory=$PWD -e '/bin/bash -c "source devel/setup.bash;rosrun local_planner_py RLLocalPlannerv2.py"'



