'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import parts_interactions, simple_box, target_shapes_sampled_points

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

import copy

from scipy.spatial.transform import Rotation as R
from .fill_volume_env_3D_multiple_obj_3Dac_more_objects2 import StackBoxesEnv3D_multiple_obj

from parts import create_complex_objects

class StackBoxesEnv3D_multiple_obj(StackBoxesEnv3D_multiple_obj):
    '''
    Env based on fill_volume_env_3D_multiple_obj_3Dac_more_objects_new2.py - now allowing more actions, i.e. 20 which
    basically add 4 rotations to the 5 placement actions
    '''

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, load_path=None, num_actions=5,disc_factor=None,fill_threshold=0.75): # for evaluation set to 0.85
        num_actions = 4*5
        self.check_stability = True
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected,
                         load_path,num_actions,disc_factor,fill_threshold)

        self.obj_creator = create_complex_objects.ComplexObjects21()
        # remove all cameras:
        self.SimEnv.remove_all_cameras()
        # define multiple cameras:
        # looking from front
        self.rotation_operations = []
        self.translation_operations = []
        self.flip_lr_operations = []
        # CAM 1:
        ref_cam_pos = np.asarray([0,-0.25,0.25])
        ref_cam_target = np.asarray([0,0.0,0.25])
        self.SimEnv.add_camera_static([0,-0.25,0.25],[0,0.0,0.25],[0,0,1],"front_cam",near=0.05,far=1.0,cam_width=100,cam_height=100)
        r = R.from_rotvec(0 * np.array([0, 0, 1]))
        r = r.as_matrix()
        t = np.asarray([0,0,0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        # CAM 2:
        r = R.from_rotvec(np.pi * np.array([0, 0, 1]))
        r = r.as_matrix()
        t = np.asarray([0,0.05,0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        self.SimEnv.add_camera_static(np.matmul(r,ref_cam_pos)+t,np.matmul(r,ref_cam_target)+t,[0,0,1],"front_cam",near=0.05,far=1.0,cam_width=100,cam_height=100)

        self.target_shapes = []

        if not(ensemble):
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0], self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0],self.rotation_operations[1], self.translation_operations[1], self.flip_lr_operations[1], width=env_grid_size,height=env_grid_size))
        else:
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[1], self.translation_operations[1], self.flip_lr_operations[1], width=env_grid_size,height=env_grid_size))
        self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)

        self.ensemble = ensemble

        self.initial_elements_covered = 0
        self.empty_env()
        self.did_rendering = False
        self.standard_image = copy.deepcopy(self._get_depth(filter=[0.0, 0.226]))

        self.populate_env()

    def decode_action(self, u, obj_list_not_placed,indices,block_to_be_placed):
        # decode action function now more complex as all parts have to be also rotated
        ac_0 = int(int(u[2]) / int(4))
        ac_1 = int(u[2]) % int(4)
        if (ac_0==0):
            idx = 4
        elif (ac_0==1):
            idx = 1
        elif (ac_0 == 2):
            idx = 0
        elif (ac_0 ==3):
            idx = 3
        else:
            idx = 2
        # ac1 encodes a rotation:
        r = R.from_rotvec(np.pi / 2 * np.array([0, 0, ac_1]))
        offset = np.asarray(self.object_list_not_placed[block_to_be_placed][:3])
        # now apply rotation also to all the other parts!
        for i in range(len(indices)):
            curr_pos = np.matmul(r.as_matrix(),np.asarray(self.object_list_not_placed[int(indices[i])][:3]) - offset)+offset
            self.object_list_not_placed[int(indices[i])][0] = curr_pos[0]
            self.object_list_not_placed[int(indices[i])][1] = curr_pos[1]
            self.object_list_not_placed[int(indices[i])][2] = curr_pos[2]
        return idx
