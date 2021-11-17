'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")

from .fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_5_4_acs import StackBoxesEnv3D_multiple_obj_w_robot as StackBoxesEnv3D_multiple_obj_w_robot_inherit

class StackBoxesEnv3D_multiple_obj_w_robot(StackBoxesEnv3D_multiple_obj_w_robot_inherit):
    '''
    This environment is built on top of fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_more_objects_5_4_acs.py and
    implements additional functionality to handle the state of the robot which is in this case also included in the
    observation
    '''

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, robot_info=True, load_path=None, complexity_int = 21, disc_factor=0.95, fill_threshold=0.75):
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected, robot_info, complexity_int = complexity_int,disc_factor=disc_factor, fill_threshold=fill_threshold)

    def preprocess_decode(self,state):
        return state[:-(2*9)]

    def preprocess_decode_multidim(self,state):
        return state[:,:-(2*9)]