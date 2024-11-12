### APP VISO

# Build docker from scratch (A)

-> Set your source mounting path first in buildandrun-sh file for the catkin_new folder: "--mount src="/home/alessandro/docker/ViSo/catkin_new"

##### (A) <Run docker and startup App>


go to folder ./buildandrun.sh

-> builds container

###open 3 terminals:

###first terminal:
-> ./buildandrun.sh
-> catkin_make
-> roslaunch ibvs ibvs.launch
=> GAZEBO RVIZ should open with object in it

###second terminal:
-> docker exec -ti appviso bash
-> cd /src/gazebo_vel_broadcaster/src
-> python3 gazebo_vel_broadcaster.py
=> Enabels the veloity broadcasting for Gazebo World

###third terminal:
-> docker exec -ti appviso bash
-> cd /src/ibvs/src
-> python3 controller.py
=> stats the controller.py Image Based Visual Servoing Method, Change goal image or try to change the position of the object in Gazebo 
    
    
    
    
    
    
NEW:

catkin_make

export GAZEBO_MODEL_PATH=~/catkin_ws/src/ibvs/models:$GAZEBO_MODEL_PATH

roslaunch ibvs ibvs.launch

cd catkin_ws/src/ibvs/src/
python3 controller_all.py
