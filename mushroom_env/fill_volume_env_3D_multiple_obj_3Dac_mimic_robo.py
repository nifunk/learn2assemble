'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import simple_box, target_shapes_sampled_points

from mushroom_rl.utils.spaces import *

import sobol_seq
import math

from scipy.spatial.transform import Rotation as R
from .fill_volume_env_3D_multiple_obj_3Dac import StackBoxesEnv3D_multiple_obj

class StackBoxesEnv3D_multiple_obj_mimic_robo(StackBoxesEnv3D_multiple_obj):

    '''
    Environment built on top of fill_volume_env_3D_multiple_obj_3Dac.py which mimics the simplest robot environment, i.e.
    the structure to be built is 1D and the blocks are initially placed in the same way as in the robot env.
    '''



    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, load_path=None):
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected)
        # remove all cameras:
        self.SimEnv.remove_all_cameras()
        # define only one single camera:
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

        self.target_shapes = []

        if not(ensemble):
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0], self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
        else:
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))

        self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)

        self.ensemble = ensemble

        self.initial_elements_covered = 0

        self.populate_env()

    def populate_env(self,default_placed=[], default_unplaced=[]):
        # have to redefine the populate_env function as we now want to place the unplaced elements in accordance with the
        # robotic environment
        if not(len(self.object_list)==0):
            self.empty_env()

        if (default_placed==[] and default_unplaced==[]):
            init_idx = np.random.choice(len(self.target_shapes))
            self.object_list.append(simple_box.Simple_box([self.translation_operations[init_idx][0], self.translation_operations[init_idx][1], 0], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            missing_samples = self.num_boxes - len(self.object_list)
            # use sobol sequences to add additional parts evenly below the scene (virtually,...)
            divider_samples = 6
            offset_y = -0.2
            num_rounds = int(math.ceil(missing_samples / divider_samples))
            total_samples = missing_samples

            for j in range(num_rounds):
                if (total_samples >= divider_samples):
                    current_samples = divider_samples
                else:
                    current_samples = total_samples
                total_samples -= current_samples
                coord_x = sobol_seq.i4_sobol_generate(1, current_samples)
                coord_x = (2 * coord_x - 1) * 0.75 * self.observation_high[1] / 2 - 0.15 * self.observation_high[1] / 2
                # place additional blocks behind the one that is placed,...
                for i in range(current_samples):
                    self.object_list_not_placed.append([coord_x[i, 0], offset_y, 0.025])
                offset_y += -0.125

        else:
            self.object_list.append(simple_box.Simple_box([default_placed[0][0], default_placed[0][1], default_placed[0][2]-0.05 / 2], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            for i in range(len(default_placed)-1):
                self.object_list.append(
                    simple_box.Simple_box([default_placed[i+1][0], default_placed[i+1][1], default_placed[i+1][2]-0.05 / 2], [0, 0, 0, 1],
                                          self.SimEnv.physicsClient))
            for i in range(len(default_unplaced)):
                self.object_list_not_placed.append([default_unplaced[i][0], default_unplaced[i][1], default_unplaced[i][2]])
