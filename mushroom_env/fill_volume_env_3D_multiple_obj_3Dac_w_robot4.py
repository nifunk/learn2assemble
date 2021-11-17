'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import parts_interactions, simple_box, target_shapes_sampled_points

from mushroom_rl.utils.spaces import *

from scipy.spatial.transform import Rotation as R
from .fill_volume_env_3D_multiple_obj_3Dac_w_robot2 import StackBoxesEnv3D_multiple_obj_w_robot as StackBoxesEnv3D_multiple_obj_w_robot_inherit

class StackBoxesEnv3D_multiple_obj_w_robot(StackBoxesEnv3D_multiple_obj_w_robot_inherit):
    '''
    4 sided robot environment - apart from cameras it inherits all functionality from fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_new_new.py
    '''



    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, robot_info=False, load_path=None):

        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected, robot_info)

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
        t = np.asarray([0,2*self.dist,0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        self.SimEnv.add_camera_static(np.matmul(r,ref_cam_pos)+t,np.matmul(r,ref_cam_target)+t,[0,0,1],"front_cam",near=0.05,far=1.0,cam_width=100,cam_height=100)
        # CAM 3:
        r = R.from_rotvec(np.pi/2 * np.array([0, 0, 1]))
        r = r.as_matrix()
        t = np.asarray([self.dist, self.dist, 0.0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        self.SimEnv.add_camera_static(np.matmul(r, ref_cam_pos) + t, np.matmul(r, ref_cam_target) + t, [0, 0, 1],
                                      "front_cam", near=0.05, far=1.0, cam_width=100, cam_height=100)
        # CAM 4:
        r = R.from_rotvec(-np.pi/2 * np.array([0, 0, 1]))
        r = r.as_matrix()
        t = np.asarray([-self.dist, self.dist, 0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        self.SimEnv.add_camera_static(np.matmul(r, ref_cam_pos) + t, np.matmul(r, ref_cam_target) + t, [0, 0, 1],
                                      "front_cam", near=0.05, far=1.0, cam_width=100, cam_height=100)
        self.target_shapes = []
        if not(ensemble):
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0], self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0],self.rotation_operations[1], self.translation_operations[1], self.flip_lr_operations[1], width=env_grid_size,height=env_grid_size))
        else:
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[1], self.translation_operations[1], self.flip_lr_operations[1], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[2], self.translation_operations[2], self.flip_lr_operations[2], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[3], self.translation_operations[3], self.flip_lr_operations[3], width=env_grid_size,height=env_grid_size))

        self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)

        self.ensemble = ensemble

        self.initial_elements_covered = 0

        self.populate_env()