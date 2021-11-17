'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import parts_interactions, simple_box, target_shapes_sampled_points
import math
import pybullet as p

from mushroom_rl.utils.spaces import *

import sobol_seq
import time
import copy

from scipy.spatial.transform import Rotation as R
from .fill_volume_env_3D_multiple_obj_3Dac_w_robot2 import StackBoxesEnv3D_multiple_obj_w_robot as StackBoxesEnv3D_multiple_obj_w_robot_inherit

from env import standart_twin


from parts import create_complex_objects

class StackBoxesEnv3D_multiple_obj_w_robot(StackBoxesEnv3D_multiple_obj_w_robot_inherit):
    '''
    Robot environments building upon fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_new_new.py, however with
    the added complexity that now the more complex objects are actually being stacked
    '''

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, robot_info=False, load_path=None, num_actions=20, check_stability=True,disc_factor=None, fill_threshold=0.75):
        # this is needed to create the new - more complex objects,...
        self.obj_creator = create_complex_objects.ComplexObjects()
        self.object_list_not_placed_info = []
        self.num_features_orig = 3  # number of features only related with pos
        # when calling the reset functions first -> dont do stability checking (i.e. if initialization of the environment
        # is stable. This is mainly targeted at making sure that no parts fall of the table as there is limited space)
        self.check_stability = False
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected, robot_info, num_actions=num_actions,disc_factor=disc_factor)
        # Exploit twin environment to make sure that the cameras checking the already built structure do not have problems
        # with potential occlusions
        self.SimEnvTwin = standart_twin.Standart(visualize=False)

        self.SimEnv.remove_all_cameras()
        self.SimEnvTwin.remove_all_cameras()
        # define multiple cameras:
        # looking from front
        self.rotation_operations = []
        self.translation_operations = []
        self.flip_lr_operations = []
        # CAM 1:
        ref_cam_pos = np.asarray([0, -0.25, 0.25])
        ref_cam_target = np.asarray([0, 0.0, 0.25])
        self.SimEnv.add_camera_static([0, -0.25, 0.25], [0, 0.0, 0.25], [0, 0, 1], "front_cam", near=0.05, far=1.0,
                                      cam_width=100, cam_height=100)
        self.SimEnvTwin.add_camera_static([0, -0.25, 0.25], [0, 0.0, 0.25], [0, 0, 1], "front_cam", near=0.05, far=1.0,
                                      cam_width=100, cam_height=100)
        r = R.from_rotvec(0 * np.array([0, 0, 1]))
        r = r.as_matrix()
        t = np.asarray([0, 0, 0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        # specify distance for the placing, i.e. how tight is the placing with respect to the other blocks
        # Note: size of a block is 0.05
        self.dist = 0.06
        # CAM 2:
        r = R.from_rotvec(np.pi * np.array([0, 0, 1]))
        r = r.as_matrix()
        t = np.asarray([0, self.dist, 0])
        fliplr = False
        self.rotation_operations.append(r)
        self.translation_operations.append(t)
        self.flip_lr_operations.append(fliplr)
        self.SimEnv.add_camera_static(np.matmul(r, ref_cam_pos) + t, np.matmul(r, ref_cam_target) + t, [0, 0, 1],
                                      "front_cam", near=0.05, far=1.0, cam_width=100, cam_height=100)
        self.SimEnvTwin.add_camera_static(np.matmul(r, ref_cam_pos) + t, np.matmul(r, ref_cam_target) + t, [0, 0, 1],
                                      "front_cam", near=0.05, far=1.0, cam_width=100, cam_height=100)

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
        # only eventually from here on do the stability checking,...
        self.check_stability = check_stability
        self.populate_env()
        self.fill_threshold=fill_threshold

    def _list_to_array_idx(self,list, idx):
        # returns the object at a specific index inside a list as an array
        obs_array = list[0][:idx]
        for i in range(len(list)-1):
            obs_array = np.concatenate((obs_array,list[i+1][:idx]))
        return obs_array

    def populate_env(self,default_placed=[], default_unplaced=[]):
        default_placed_backup = copy.deepcopy(default_placed)
        default_unplaced_backup = copy.deepcopy(default_unplaced)
        # if env not already empty - do this
        if not(len(self.object_list)==0):
            self.empty_env()

        if (default_placed==[] and default_unplaced==[]):
            # in case no desired set of already placed and unplaced is provided -> first sample a blockset
            list_of_objects = self.obj_creator.sample_blockset(self.num_boxes)
            # first place one initial block at a random side of the environment
            init_idx = np.random.choice(len(self.target_shapes))
            self.object_list.append(simple_box.Simple_box([self.translation_operations[init_idx][0], self.translation_operations[init_idx][1], 0], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])

            self.list_all_obj = []
            dimensions = self.object_list[0].dimensions
            dimensions[0] = dimensions[1] = self.dist / 2
            for i in range(len(list_of_objects)):
                for j in range(len(list_of_objects[i])):
                    # modify the list of objects from the obj creator by adding the right scale, as it only returned
                    # the relative positions of the blocks
                    list_of_objects[i][j][:3] = (
                        np.multiply(np.asarray(list_of_objects[i][j][:3]), 2 * np.asarray(dimensions))).tolist()
                    self.list_all_obj.append(list_of_objects[i][j])
            # NOW POSITION THE FIRST OBJECT CORRECTLY (entire object)
            parts_interactions.stack_complex_parts_primitively_absolute(self.object_list[0], None, self.list_all_obj,
                                                                        self.object_list, self.SimEnv.physicsClient)
            # determine how many blocks are still to be placed
            missing_samples = self.obj_creator.get_num_remaining_blocks(self.list_all_obj)
            # use sobol sequences to add additional parts evenly & try to make sure that they fit on the table
            divider_samples = 6
            offset_y = -0.15
            offset_x_1 = -0.0
            num_rounds = int(math.ceil(missing_samples/divider_samples))
            total_samples = missing_samples
            for j in range(num_rounds):
                if (total_samples>=divider_samples):
                    current_samples = divider_samples
                else:
                    current_samples = total_samples
                total_samples -= current_samples
                coord_x = sobol_seq.i4_sobol_generate(1,current_samples)
                coord_x = (2*coord_x - 1)*0.75*self.observation_high[1]/2 - 0.15*self.observation_high[1]/2
                coord_x[:,0] += offset_x_1
                if (j==0):
                    coord_x_final = coord_x
                else:
                    coord_x_final = np.vstack((coord_x_final,coord_x))
            # given the sampled coordinates -> actually place the blocks
            self.obj_creator.place_unplaced(self.list_all_obj, self.object_list_not_placed_info, coord_x_final,
                                            [0, offset_y, 0.0])
            unplaced_as_arr = self._list_to_array_idx(self.object_list_not_placed_info, 3).reshape(-1, 3)
            # check x,y,z coords of all the unplaced blocks and ensure that they are inside the bounds of our observation
            check_x_coord = np.sum((np.logical_and(unplaced_as_arr[:, 0] > self.observation_low[0],
                                                   unplaced_as_arr[:, 0] < self.observation_high[0])) - 1)
            check_y_coord = np.sum((np.logical_and(unplaced_as_arr[:, 1] > self.observation_low[1],
                                                   unplaced_as_arr[:, 1] < self.observation_high[1])) - 1)
            check_z_coord = np.sum((np.logical_and(unplaced_as_arr[:, 2] > self.observation_low[2],
                                                   unplaced_as_arr[:, 2] < self.observation_high[2])) - 1)
            if ((check_x_coord + check_y_coord + check_z_coord) != 0):
                print("CAUTION: ONE OF THE PARTS IS PLACED OUT OF BOUNDS,....")

        else:
            # simply place all objects that are already placed
            self.object_list.append(simple_box.Simple_box([default_placed[0][0], default_placed[0][1], default_placed[0][2]-0.05 / 2], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            for i in range(len(default_placed)-1):
                self.object_list.append(
                    simple_box.Simple_box([default_placed[i+1][0], default_placed[i+1][1], default_placed[i+1][2]-0.05 / 2], [0, 0, 0, 1],
                                          self.SimEnv.physicsClient))
            # before placing the unplaced blocks - first have to check if they are close to each other to figure out if
            # they belong together or not,...
            dimensions = self.object_list[0].dimensions
            dimensions[0] = dimensions[1] = self.dist / 2
            unplaced_elements_list = self.obj_creator.decompose_unplaced(copy.deepcopy(default_unplaced), 2*np.asarray(dimensions))
            for i in range(len(default_unplaced)):
                self.object_list_not_placed_info.append([*unplaced_elements_list[i]])
            # for all unplaced elements consisting of more than 1 primitive object, make sure that they are sorted
            for i in range(len(self.object_list_not_placed_info)):
                if (len(self.object_list_not_placed_info[i][3:])>1):
                    sublist = copy.deepcopy(self.object_list_not_placed_info[i][3:])
                    sublist.sort()
                    for j in range(len(self.object_list_not_placed_info[i][3:])):
                        self.object_list_not_placed_info[i][3+j] = sublist[j]
        # now place the unplaced elements
        for i in range(len(self.object_list_not_placed_info)):
            min_previous = np.min(self.object_list_not_placed_info[i][3:])
            # if min_previous is 0 -> this is the first primitive part of the object -> simply place it here
            if (min_previous == 0):
                self.object_list_not_placed.append(
                    simple_box.Simple_box(
                        [self.object_list_not_placed_info[i][0], self.object_list_not_placed_info[i][1],
                         0.0 + self.object_list_not_placed_info[i][2]],
                        [0, 0, 0, 1],
                        self.SimEnv.physicsClient))
            else:
                # otherwise -> have to place it but place it such that it takes into account the constraints to the other
                # primitive parts of the object - mainly here we have to make sure there is enough spacing to the other blocks
                tmp_list = self.object_list_not_placed_info[i+int(min_previous):i]
                unplaced_related_arr = np.asarray(self._list_to_array_idx(tmp_list,3)).reshape(-1,3)
                dimensions = self.object_list[0].dimensions
                dimensions[0] = dimensions[1] = self.dist / 2
                self.object_list_not_placed.append(
                    simple_box.Simple_box_with_surrounding(
                        [self.object_list_not_placed_info[i][0], self.object_list_not_placed_info[i][1],
                         0.0 + self.object_list_not_placed_info[i][2]],
                        [0, 0, 0, 1],
                        self.SimEnv.physicsClient,2*np.asarray(dimensions),unplaced_related_arr,0.01))
            # if there are other primitive blocks already placed, then this is !=0 (as this list has been sorted
            # previously). then we have to create constraints with the other blocks
            if (self.object_list_not_placed_info[i][3]!=0.0):
                lower_obj = self.object_list_not_placed[-1+int(self.object_list_not_placed_info[i][3])]
                upper_obj = self.object_list_not_placed[-1]
                cid = p.createConstraint(lower_obj.handle, -1, upper_obj.handle, -1, p.JOINT_FIXED, [0, 0, 0],
                                         upper_obj.get_pos_orient()[0] - lower_obj.get_pos_orient()[0],
                                         [0, 0, 0.0], physicsClientId=lower_obj.physicsClient)
        # only do stability check if indicated by variable and if we sampled new scene, i.e. the default_unplaced_backup
        # and the other one are empty,...
        if (self.check_stability and (default_unplaced_backup==[]) and (default_placed_backup==[])):
            # check stability of structure:
            stability = self.SimEnv.check_initial_stability(self.object_list,self.object_list_not_placed)
            if (not(stability) and (default_unplaced_backup==[]) and (default_placed_backup==[])):
                # if stability criterion is not fullfilled -> we sample anwe
                print ("populate again")
                self.populate_env(default_unplaced_backup,default_placed)

    def decompose_action(self,u):
        # function that decomposes the action
        p_offsets = []
        grasps = []
        # decompose action - there are 5 placement actions and 4 grasps
        # offsets list contains the waypoints that are to be reached to place the part properly
        # grasps defines the grasping and placing poses - this set of grasping and placing also directly specifies a potential
        # relative rotation of the part
        ac_0 = int(int(u[2]) / int(4))
        ac_1 = int(u[2]) % int(4)
        dist = self.dist
        if (ac_0 == 0):
            idx = 4
            # on top:
            p_offsets.append([0, 0, 0.25])
            p_offsets.append([0, 0, 0.075])
            p_offsets.append([0, 0, 0.055])
            p_offsets.append([0, 0, 0.05 + 0.005])
        elif (ac_0 == 1):
            idx = 1
            p_offsets.append([-dist, 0, 0.25])
            p_offsets.append([-dist, 0, 0.075 - 0.05])
            p_offsets.append([-dist, 0, 0.055 - 0.05 + 0.005])
            p_offsets.append([-dist, 0, 0.05 - 0.05 + 0.005])
        elif (ac_0 == 2):
            idx = 0
            p_offsets.append([dist, 0, 0.25])
            p_offsets.append([dist, 0, 0.075 - 0.05])
            p_offsets.append([dist, 0, 0.055 - 0.05 + 0.005])
            p_offsets.append([dist, 0, 0.05 - 0.05 + 0.005])
        elif (ac_0 == 3):
            idx = 3
            p_offsets.append([0, -dist, 0.25])
            p_offsets.append([0, -dist, 0.075 - 0.05])
            p_offsets.append([0, -dist, 0.055 - 0.05 + 0.005])
            p_offsets.append([0, -dist, 0.05 - 0.05 + 0.005])
        else:
            idx = 2
            p_offsets.append([0, dist, 0.25])
            p_offsets.append([0, dist, 0.075 - 0.05])
            p_offsets.append([0, dist, 0.055 - 0.05 + 0.005])
            p_offsets.append([0, dist, 0.05 - 0.05 + 0.005])
        # select the grasps here
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
        # find out number of objects in the scene before adding the new one
        curr_num_targets = len(self.target_shape_list)
        curr_num_obj = len(self.object_list)
        curr_num_objects_not_placed = len(self.object_list_not_placed)
        # determine which block should be placed - i.e. decode the action
        block_to_be_placed = np.clip(u[0], curr_num_targets + curr_num_obj,
                                     curr_num_targets + curr_num_obj + curr_num_objects_not_placed - 1) - (
                                         curr_num_targets + curr_num_obj)
        reference_block = np.clip(u[1], curr_num_targets, curr_num_targets + curr_num_obj - 1) - (curr_num_targets)
        # this is the object that is to be moved
        obj_to_be_moved =  self.object_list_not_placed[block_to_be_placed]
        pos, ori = obj_to_be_moved.get_pos_orient()
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
        successfull = self.SimEnv.grasp_part_w_vel_check(pos, grasps[0], obj_to_be_moved, self.object_list_not_placed,invalid_idx)
        # if grasping was successfull
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
            # actually do placing of the part
            successfull = self.SimEnv.place_part_w_table_collision_check(pos, p_offsets, grasps[1], lower_obj, obj_to_be_moved,
                                                 self.object_list,self.object_list_not_placed,invalid_idx,grasp_bias=grasp_bias)
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

        # finally transfer ALL (could be multiple) of the parts (primitive elements) into the placed elements
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
        finished = False
        # if no more objects available - finish
        if (len(self.object_list_not_placed) == 0):
            finished = True

        # compute the reward
        placed_parts_obs = self._list_to_array(self._get_list_placed_elements())
        rew, finished1, terminal_rew = self._get_reward(placed_parts_obs)
        # get the observation
        obs = self._get_obs()
        # if invalid action - return penalty of -1 ; else give additional reward of +1 of successful
        if (prior_exit):
            rew = -1.0
        else:
            #i.e. no prior exit
            # but if completed succesfully -> add additional reward of +1
            if (finished1):
                rew += 1.0

        return obs, rew, (finished or finished1 or prior_exit), {}

    def _get_obs(self):
        # get the observation
        target_elements = self.target_shape_list
        real_elements = self._get_list_placed_elements()
        unplaced_elements = self.object_list_not_placed_info
        obs = self._encode_observation(target_elements,real_elements,unplaced_elements)
        return obs

    def _encode_observation(self,target_elements,real_elements,unplaced_elements):
        add_array = np.asarray([])
        if self.add_connectivity:
            add_array = np.ones(self.total_num_nodes)
            add_array_existing_blocks = copy.deepcopy(add_array)
            if (not self.fully_connected):
                start_unplaced_elem = len(target_elements)+len(real_elements)
                add_array[start_unplaced_elem:] = 0

        total_counter=0
        obs_array = np.concatenate((np.asarray(target_elements[0][:self.num_features_orig]),np.asarray([1,1]),add_array_existing_blocks))
        total_counter += 1
        for i in range(len(target_elements)-1):
            obs_array1 =np.concatenate((np.asarray(target_elements[i+1][:self.num_features_orig]),np.asarray([1,1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1
        for i in range(len(real_elements)):
            obs_array1 =np.concatenate((np.asarray(real_elements[i][:self.num_features_orig]),np.asarray([1,-1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1
        # for all unplaced elements - in partially connected case - create special connectivity
        for i in range(len(unplaced_elements)):
            if (not self.fully_connected):
                connected_elements = np.asarray(unplaced_elements[i][3:])
                connected_elements = (connected_elements + total_counter).astype(int)
                new_arr = copy.deepcopy(add_array)
                new_arr[connected_elements] = 1.0
                obs_array1 =np.concatenate((np.asarray(unplaced_elements[i][:self.num_features_orig]),np.asarray([-1,-1]),new_arr))
            else:
                obs_array1 =np.concatenate((np.asarray(unplaced_elements[i][:self.num_features_orig]),np.asarray([-1,-1]),add_array_existing_blocks))

            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1
        # potentially add info of the robot to the observation
        if (self.robot_info):
            robot_pos, robot_vel = self.SimEnv.get_robot_info()
            obs_array = np.concatenate((obs_array, np.asarray(robot_pos)))
            obs_array = np.concatenate((obs_array, np.asarray(robot_vel)))
        return obs_array

    def _get_reward(self,obs):
        # calculate the reward signal
        elem_covered = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_rew1 = ((elem_covered)/(self.total_elements))
        #Calculate everything with holistic view again:
        elem_covered_complete = np.sum(np.multiply(self._get_depth(filter=[0.0,0.25]), self.target_shape_mask))
        curr_rew_complete = ((elem_covered_complete)/(self.total_elements))
        return_rew = curr_rew1 - self.prev_rew

        self.prev_rew = curr_rew1
        unified_area = np.sum(np.clip(self._get_depth() + self.target_shape_mask, 0, 1))
        intersecton = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_iou = intersecton / unified_area
        diff_iou = curr_iou - self.prev_iou
        self.prev_iou = curr_iou
        # we return the difference, i.e. by how much the filling has changed, however, clipped with 0
        if (return_rew<=0):
            return_rew = 0.0
        else:
            return_rew = return_rew

        return 3*return_rew, (curr_rew_complete>self.fill_threshold), curr_rew1

    def _get_env_info(self):
        # get the info of the environment
        elem_covered = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_rew1 = ((elem_covered)/(self.total_elements))
        #Calculate everything with holistic view again:
        elem_covered_complete = np.sum(np.multiply(self._get_depth(filter=[0.0,0.25]), self.target_shape_mask))
        curr_rew_complete = ((elem_covered_complete)/(self.total_elements))
        return_rew = curr_rew1 - self.prev_rew

        self.prev_rew = curr_rew1
        unified_area = np.sum(np.clip(self._get_depth() + self.target_shape_mask, 0, 1))
        intersecton = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_iou = intersecton / unified_area
        diff_iou = curr_iou - self.prev_iou
        self.prev_iou = curr_iou
        if (return_rew<=0):
            # return_rew = -1.0/self.total_num_nodes
            return_rew = 0.0
        else:
            return_rew = return_rew
        finished = False
        if (len(self.object_list_not_placed)==0):
            finished = True
        return return_rew, curr_iou, (curr_rew_complete > self.fill_threshold), finished

    def empty_env(self):
        # empty environment, i.e. remove all objects
        for i in range(len(self.object_list)):
            self.object_list[i].remove()
        for i in range(len(self.object_list_not_placed)):
            self.object_list_not_placed[i].remove()
        self.object_list.clear()
        self.object_list_not_placed.clear()
        self.object_list_not_placed_info.clear()

    # have to override this function as we are now using "digital twin"
    def _get_depth(self,filter=None):
        if not(self.did_rendering):
            self.SimEnvTwin.render(self.object_list)
            self.did_rendering = True
            self.raw_image = []
            for i in range(len(self.target_shapes)):
                self.raw_image.append(copy.deepcopy(self.SimEnvTwin.get_depth(cam_id=i)))

        self.image = []
        for i in range(len(self.target_shapes)):
            if (filter is None):
                self.image.append(self._do_filtering(copy.deepcopy(self.raw_image[i]),filter=[0.21,0.25]))
            else:
                self.image.append(self._do_filtering(copy.deepcopy(self.raw_image[i]), filter=filter))
        return self.image

