'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import parts_interactions, simple_box, target_shapes_sampled_points
from mushroom_rl.utils.spaces import *

import time
import copy

from .fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2 import StackBoxesEnv3D_multiple_obj_w_robot as StackBoxesEnv3D_multiple_obj_w_robot_inherit

from parts import create_complex_objects

class StackBoxesEnv3D_multiple_obj_w_robot(StackBoxesEnv3D_multiple_obj_w_robot_inherit):
    '''
    This environment builds upon the fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_more_objects.py
    Expands it by adding so called height_offset which puts the parts (primitive blocks) at the correct height
    '''

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, robot_info=False, load_path=None, num_actions=5*4, complexity_int = 0, check_stability=False,disc_factor=None, fill_threshold=0.75):
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected, robot_info, num_actions=num_actions, check_stability=check_stability,disc_factor=disc_factor, fill_threshold=fill_threshold)
        self.target_shapes = []
        # decide on which objects are to be placed
        if (complexity_int == 0):
            self.obj_creator = create_complex_objects.ComplexObjects()
        elif (complexity_int == 1):
            self.obj_creator = create_complex_objects.ComplexObjects1()
        elif (complexity_int == 2):
            self.obj_creator = create_complex_objects.ComplexObjects2()
        elif (complexity_int == 3):
            self.obj_creator = create_complex_objects.ComplexObjects3()
        elif (complexity_int == 4):
            self.obj_creator = create_complex_objects.ComplexObjects4()
        elif (complexity_int == 20):
            self.obj_creator = create_complex_objects.ComplexObjects20()
        elif (complexity_int == 21):
            self.obj_creator = create_complex_objects.ComplexObjects21()
        elif (complexity_int == 23):
            self.obj_creator = create_complex_objects.ComplexObjects23()
        elif (complexity_int == 25):
            self.obj_creator = create_complex_objects.ComplexObjects25()
        elif (complexity_int == 26):
            self.obj_creator = create_complex_objects.ComplexObjects26()
        self.target_shapes = []
        self.check_stability = True

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

        self.check_stability = True
        self.slow_down_env = False
        self.populate_env()

    def decompose_action(self,u):
        # decompose the action into a basically placement and grasp / place pose action
        p_offsets = []
        grasps = []
        # print (u[2])
        ac_0 = int(int(u[2]) / int(4))
        ac_1 = int(u[2]) % int(4)
        dist = self.dist
        if (ac_0==0):
            idx = 4
            # on top:
            p_offsets.append([0,0,0.05 + 0.25])
            p_offsets.append([0,0,0.05 + 0.075-0.05])
            p_offsets.append([0,0,0.05 + 0.055 - 0.05 + 0.005])
            p_offsets.append([0,0,0.05 + 0.05 - 0.05 + 0.005])
        elif (ac_0==1):
            idx = 1
            p_offsets.append([-dist,0,0.25])
            p_offsets.append([-dist,0,0.075-0.05])
            p_offsets.append([-dist,0,0.055-0.05+0.005])
            p_offsets.append([-dist,0,0.05-0.05+0.005])
        elif (ac_0 == 2):
            idx = 0
            p_offsets.append([dist,0,0.25])
            p_offsets.append([dist,0,0.075-0.05])
            p_offsets.append([dist,0,0.055-0.05+0.005])
            p_offsets.append([dist,0,0.05-0.05+0.005])
        elif (ac_0 ==3):
            idx = 3
            p_offsets.append([0,-dist,0.25])
            p_offsets.append([0,-dist,0.075-0.05])
            p_offsets.append([0,-dist,0.055-0.05+0.005])
            p_offsets.append([0,-dist,0.05-0.05+0.005])
        else:
            idx = 2
            p_offsets.append([0,dist,0.25])
            p_offsets.append([0,dist,0.075-0.05])
            p_offsets.append([0,dist,0.055-0.05+0.005])
            p_offsets.append([0,dist,0.05-0.05+0.005])
        if (ac_1 == 0):
            grasps.append([-1, -0., 0., 0.])
            grasps.append([-1, -0., 0., 0.])
        if (ac_1 == 1):
            grasps.append([-1, -0., 0., 0.])
            grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
        if (ac_1 == 2):
            grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
            grasps.append([-1, -0., 0., 0.])
        if (ac_1 == 3):
            grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
            grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
        return p_offsets, grasps, None

    def step(self, u):
        self.did_rendering = False
        if (self.visualize):
            self.visualize_target_shape_multiple()
            time.sleep(0.5)

        self.curr_time += 1
        # find out number of objects before adding the new one, i.e. decompose state of environment
        curr_num_targets = len(self.target_shape_list)
        curr_num_obj = len(self.object_list)
        curr_num_objects_not_placed = len(self.object_list_not_placed)
        # determine which block should be placed, i.e. decompose the action
        block_to_be_placed = np.clip(u[0], curr_num_targets + curr_num_obj,
                                     curr_num_targets + curr_num_obj + curr_num_objects_not_placed - 1) - (
                                         curr_num_targets + curr_num_obj)
        reference_block = np.clip(u[1], curr_num_targets, curr_num_targets + curr_num_obj - 1) - (curr_num_targets)
        # this is the object that is to be moved
        obj_to_be_moved =  self.object_list_not_placed[block_to_be_placed]
        # read out the height offset which is crucial for the correct placement of the part
        pos, ori = obj_to_be_moved.get_pos_orient()
        offset_height = np.asarray(copy.deepcopy(self.object_list_not_placed_info[block_to_be_placed][:3]))
        offset_height[0] = offset_height[1] = 0.0
        # determine the action (go from encoding to actual one)
        p_offsets, grasps, grasp_bias = self.decompose_action(u)
        # retrieve the indices of the blocks that are connected to the one that is being placed
        invalid_idx = np.asarray(self.object_list_not_placed_info[block_to_be_placed][3:]) + block_to_be_placed
        # before grasping retrieve positions of blocks that are not placed yet (including the block that is to be placed)
        curr_pos_list_unplaced = []
        curr_pos_to_be_placed = []
        for i in range(len(self.object_list_not_placed)):
            if (not(np.any(invalid_idx==i))):
                curr_pos_list_unplaced.append(self.object_list_not_placed[i].get_pos_orient()[0])
            else:
                curr_pos_to_be_placed.append(self.object_list_not_placed[i].get_pos_orient()[0])
        if (len(curr_pos_list_unplaced)!=0):
            curr_pos_list_unplaced = np.asarray(curr_pos_list_unplaced).reshape(-1, 3)
        curr_pos_to_be_placed = np.asarray(curr_pos_to_be_placed).reshape(-1, 3)
        # now do the grasping
        successfull = self.SimEnv.grasp_part_w_vel_check(pos, grasps[0], obj_to_be_moved, self.object_list_not_placed,invalid_idx,slow_down=(self.visualize and self.slow_down_env))
        if (successfull):
            # RetrievCurrent positions of placed parts:
            curr_pos_list = []
            for i in range(len(self.object_list)):
                curr_pos_list.append(self.object_list[i].get_pos_orient()[0])
            curr_pos_list = np.asarray(curr_pos_list).reshape(-1, 3)

        self.object_list.append(obj_to_be_moved)
        lower_obj = self.object_list[reference_block]
        pos, ori = lower_obj.get_pos_orient()
        if (successfull):
            # actually do place the part
            successfull = self.SimEnv.place_part_w_table_collision_check(np.asarray(pos)+offset_height, p_offsets, grasps[1], lower_obj, obj_to_be_moved,
                                                 self.object_list,self.object_list_not_placed,invalid_idx,grasp_bias=grasp_bias,slow_down=(self.visualize and self.slow_down_env))

        if (successfull):
            # check if the placed part is still in the same direction as before ('in terms of height as orientation might
            # change due to rotation of the grasp')
            if (len(invalid_idx)!=1):
                curr_pos_to_be_placed_updated = []
                # can only do this check if more than one object
                for i in range((len(invalid_idx))):
                    curr_pos_to_be_placed_updated.append(self.object_list_not_placed[invalid_idx[i].astype(int)].get_pos_orient()[0])
                curr_pos_to_be_placed_updated = np.asarray(curr_pos_to_be_placed_updated).reshape(-1, 3)
                difference = np.abs(np.subtract(curr_pos_to_be_placed,curr_pos_to_be_placed_updated))[:,2]
                decider = np.abs(difference-np.max(difference))
                if (np.any(decider>0.001)):
                    successfull = False
                    # print ("OREITNATION MUST HAVE CHANGED,...")

        # finally transfer ALL (could be multiple) of the primitive blocks of an object into the placed elements
        invalid_idx = np.sort(invalid_idx)
        for i in range(len(invalid_idx)):
            counter = -i-1
            obj = self.object_list_not_placed.pop(int(invalid_idx[counter]))
            if (invalid_idx[counter] != block_to_be_placed):
                # if not the block to be placed -> still has to be added to the list
                self.object_list.append(obj)
            # also remove block from the info list (this applies to all)
            self.object_list_not_placed_info.pop(int(invalid_idx[counter]))

        prior_exit = False
        if (successfull):
            # only then check additionally if unplaced parts might have moved:
            curr_pos_list_unplaced_updated = []
            for i in range(len(self.object_list_not_placed)):
                curr_pos_list_unplaced_updated.append(self.object_list_not_placed[i].get_pos_orient()[0])
            if (len(curr_pos_list_unplaced)!=0):
                # Calculate distances:
                curr_pos_list_unplaced_updated = np.asarray(curr_pos_list_unplaced_updated).reshape(-1, 3)
                # Take difference
                distance_arr = np.abs(np.subtract(curr_pos_list_unplaced_updated, curr_pos_list_unplaced)) ** 2
                dist_compressed = np.sqrt(distance_arr[:, 0] + distance_arr[:, 1])
                max_movement = np.max(dist_compressed)
                if (max_movement >= 0.01):
                    successfull = False

            if (successfull):
                # only then check additionally if parts might have moved (except for the ones that have been placed,...)
                curr_pos_list_updated = []
                for i in range(len(self.object_list) - len(invalid_idx)):
                    curr_pos_list_updated.append(self.object_list[i].get_pos_orient()[0])
                # Calculate distances:
                curr_pos_list_updated = np.asarray(curr_pos_list_updated).reshape(-1, 3)
                # Take difference
                distance_arr = np.abs(np.subtract(curr_pos_list, curr_pos_list_updated)) ** 2
                dist_compressed = np.sqrt(distance_arr[:, 0] + distance_arr[:, 1])
                max_movement = np.max(dist_compressed)
                if (max_movement >= 0.01):
                    successfull = False

        if (not (successfull)):
            prior_exit = True

        # check if there are more blocks remaining in the environment after this placing action
        finished = False
        if (len(self.object_list_not_placed) == 0):
            finished = True

        # special information needed to compute the reward
        placed_parts_obs = self._list_to_array(self._get_list_placed_elements())
        rew, finished1, terminal_rew = self._get_reward(placed_parts_obs)
        # get the observation
        obs = self._get_obs()
        # determine additional rewards in case of invalid action / successful completion of the episode
        if (prior_exit):
            rew = -1.0
        else:
            if (finished1):
                rew += 1.0
        return obs, rew, (finished or finished1 or prior_exit), {}