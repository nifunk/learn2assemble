This folder contains the implementation of all the environments described in the
following:

* **fill_volume_env_3D_multiple_obj_3Dac** 2-sided environment for stacking simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_4** 4-sided environment for stacking simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_mimic_robo** 1-sided environment for stacking simple boxes
Note: the boxes are placed in the same way in which they are placed for the robot environments
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects** 1-sided environment with the more complex objects but still
only with the 5 actions (place top / left / right / front / behind)
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects2** 2-sided environment with the more complex objects and 5
actions
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects2_20acs** 2-sided environment with the more complex objects and
20 available actions -> also allowing to rotate parts
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2** 2-sided robotic environment with the 
complex objects
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_5_4_acs** 2-sided robotic environment with the 
complex objects and the correct 20-dimensional action space
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_5_4_acs_robust** 2-sided robotic environment 
with the complex objects and the correct 20-dimensional action space. Also provides a warm start for the IK solution
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_5_4_acs_robust_lr** 2-sided robotic environment 
with the complex objects and the correct 20-dimensional action space. Also provides a warm start for the IK solution and
allows to place the parts at arbitrary positions of the table
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_wrap_robot** environment including the robot observation for the
2-sided environment with the more complex objects and warm starting the IK solution
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_wrap_robot_5_4** environment including the robot 
observation for the 2-sided environment with the more complex objects
* **fill_volume_env_3D_multiple_obj_3Dac_w_robot1** 1-sided robotic environment with the simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2** 2-sided robotic environment with the simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot4** 4-sided robotic environment with the simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_wrap_robot** environment including the robot observation for the
2-sided environment for stacking simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_wrap_robot_1** environment including the robot observation for the
1-sided environment for stacking simple boxes
* **fill_volume_env_3D_multiple_obj_3Dac_wrap_robot_4** environment including the robot observation for the
4-sided environment for stacking simple boxes