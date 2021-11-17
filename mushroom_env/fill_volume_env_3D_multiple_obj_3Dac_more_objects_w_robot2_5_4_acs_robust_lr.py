'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")

from parts import parts_interactions, simple_box, target_shapes_sampled_points

from mushroom_rl.utils.spaces import *

import time
import copy

from .fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_5_4_acs import StackBoxesEnv3D_multiple_obj_w_robot as StackBoxesEnv3D_multiple_obj_w_robot_inherit

from parts import create_complex_objects

class StackBoxesEnv3D_multiple_obj_w_robot(StackBoxesEnv3D_multiple_obj_w_robot_inherit):
    '''
    This environment builds upon the fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_more_objects_5_4_acs.py
    one but stabillizes calculation of IK by providing warm starts for all of the poses!
    Compared to fill_volume_env_3D_multiple_obj_3Dac_more_objects_w_robot2_more_objects_5_4_acs_robust.py it is further
    extended for proper placing if the blocks are placed on either side of the manipulator of the scene
    '''

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, robot_info=False, load_path=None, num_actions=5*4, complexity_int = 4, disc_factor=None, fill_threshold=0.75):
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected, robot_info, num_actions=num_actions,disc_factor=disc_factor, fill_threshold=fill_threshold)
        self.target_shapes = []
        # decide which set of objects to be used
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
        # NOTE: this variable slows down the speed of the simulation in the visualized environment which might be useful
        # for obtaining nice videos
        self.slow_down_env = True
        self.populate_env()

    def decompose_action(self,u, position):
        # decompose the action
        p_offsets = []
        grasps = []
        grasp_bias = None
        # print (u[2])
        ac_0 = int(int(u[2]) / int(4))
        ac_1 = int(u[2]) % int(4)
        dist = self.dist
        # determine relative placing (top / left/ right / front/ behind) through waypoints
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
        # determine the grasping and placing posese - additionally add the so-called "grasp bias" to warm start the IK solver
        # to ensure to reach the desired grasping and placing poses.
        # Note: compared to the other envs this bias is now even dependent on the position that we want to reach to
        # even achieve better robustness
        grasp_bias = []
        if (ac_1 == 0):
            grasps.append([-1, -0., 0., 0.])
            grasps.append([-1, -0., 0., 0.])
            grasp_bias.append([0.49042033303553495, -0.5004927848780336, 0.1022805485240877, -2.7083697274369505, 0.06104799184126972, 2.2091870955566284, 1.3292189665664436, 0.06, 0.06])
            grasp_bias.append([0.10301486190478716, 0.3710846065527934, 0.025667321814051545, \
            -2.551506767014544, -0.04486787989897513, 2.923677724381635, 0.9485037918976392, 0.08, 0.08])
        if (ac_1 == 1):
            grasps.append([-1, -0., 0., 0.])
            grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
            grasp_bias.append([0.49042033303553495, -0.5004927848780336, 0.1022805485240877, -2.7083697274369505, 0.06104799184126972, 2.2091870955566284, 1.3292189665664436, 0.06, 0.06])
            grasp_bias.append([-0.16678471514757157, 0.39032691991337487, 0.22726008377285442, -2.543004065259898, \
            -0.3584032358630825, 2.922273527177446, 2.7669423707063614, 0.08, 0.08])
        if (ac_1 == 2):
            if (position[0]>-0.2 and position[1]>0.0 or position[1]<0.0):
                grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
                grasps.append([-1, -0., 0., 0.])
                grasp_bias.append([0.5401618569574408, 0.20533149020772257, -0.05006493309311179, -2.230651829425065, 0.015017246330438477, 2.436558599834874, 2.8357188164736544, 0.06, 0.06])
                grasp_bias.append([0.23485225916776117, 0.3726321091331324, -0.07944742413759208, -2.5509261077011707, \
                0.1233986336560716, 2.923508544947951, 0.8360336682853126, 0.08, 0.08])
            else:
                grasps.append([0.7071754336357117, 0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
                grasps.append([0., 1.0, 0., 0.])
                grasp_bias.append([0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.06, 0.06])
                grasp_bias.append([-0.40586779835432185, -0.13442292267676798, 0.7450972460241291, -2.7281354634046386,
                              0.18581207727087934, 2.622642866566514, -2.1816624213908633, 0.08, 0.08])
        if (ac_1 == 3):
            if (position[0] > -0.2 and position[1] > 0.0 or position[1] < 0.0):
                grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
                grasps.append([0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
                grasp_bias.append([0.5401618569574408, 0.20533149020772257, -0.05006493309311179, -2.230651829425065, 0.015017246330438477, 2.436558599834874, 2.8357188164736544, 0.06, 0.06])
                grasp_bias.append([0.06518543632966166, 0.3777420589390796, 0.0530928159042723, -2.5577630109930936, \
                -0.08844979349405321, 2.9479927605584044, 2.558375662644588, 0.08, 0.08])
            else:
                grasps.append([0.7071754336357117, 0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
                grasps.append([0.7071754336357117, 0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254])
                grasp_bias.append([0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.06, 0.06])
                grasp_bias.append([-0.1669536098014562, 0.24455027616848493, 0.1430438343350627, -2.392795939019828, -0.10573865315396437, \
                     2.745288936840627, -0.7228100794665633, 0.08, 0.08])
        return p_offsets, grasps, grasp_bias

    def step(self, u):
        # overwriting the step function is needed as we now additionally give a bias on the grasping!
        self.did_rendering = False

        if (self.visualize):
            self.visualize_target_shape_multiple()
            time.sleep(0.5)

        self.curr_time += 1
        # find out number of objects before adding the new one:
        curr_num_targets = len(self.target_shape_list)
        curr_num_obj = len(self.object_list)
        curr_num_objects_not_placed = len(self.object_list_not_placed)
        # determine which block should be place - decompose the action
        block_to_be_placed = np.clip(u[0], curr_num_targets + curr_num_obj,
                                     curr_num_targets + curr_num_obj + curr_num_objects_not_placed - 1) - (
                                         curr_num_targets + curr_num_obj)
        reference_block = np.clip(u[1], curr_num_targets, curr_num_targets + curr_num_obj - 1) - (curr_num_targets)
        # this is the object that is to be moved
        obj_to_be_moved =  self.object_list_not_placed[block_to_be_placed]
        pos, ori = obj_to_be_moved.get_pos_orient()
        # add height offset for correct placing
        offset_height = np.asarray(copy.deepcopy(self.object_list_not_placed_info[block_to_be_placed][:3]))
        offset_height[0] = offset_height[1] = 0.0
        # determine the action (go from encoding to actual one)
        p_offsets, grasps, grasp_bias = self.decompose_action(u,pos)
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
        if (pos[1]>0 and len(grasp_bias)!=0):
            # if being on the other side:
            initial_pose = grasp_bias[0]
        else:
            initial_pose = None

        if (len(grasp_bias)!=0):
            successfull = self.SimEnv.grasp_part_w_vel_check(pos, grasps[0], obj_to_be_moved, self.object_list_not_placed,invalid_idx,initial_pose=initial_pose,slow_down=(self.visualize and self.slow_down_env))
        else:
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
            if (len(grasp_bias)!=0):
                successfull = self.SimEnv.place_part_w_table_collision_check(np.asarray(pos)+offset_height, p_offsets, grasps[1], lower_obj, obj_to_be_moved,
                                                     self.object_list,self.object_list_not_placed,invalid_idx,grasp_bias=grasp_bias[1],slow_down=(self.visualize and self.slow_down_env))
            else:
                successfull = self.SimEnv.place_part_w_table_collision_check(np.asarray(pos)+offset_height, p_offsets, grasps[1], lower_obj, obj_to_be_moved,
                                                     self.object_list,self.object_list_not_placed,invalid_idx,slow_down=(self.visualize and self.slow_down_env))
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

        # finally transfer ALL (could be multiple) of the primitive elements belonging to the part into the placed elements
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
        add_velocity_penalty = 0.0
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
                # Take difference but only in x-y plane
                distance_arr = np.abs(np.subtract(curr_pos_list, curr_pos_list_updated)) ** 2
                dist_compressed = np.sqrt(distance_arr[:, 0] + distance_arr[:, 1])
                max_movement = np.max(dist_compressed)
                if (max_movement >= 0.01):
                    successfull = False

        if (not (successfull)):
            prior_exit = True
        finished = False
        # if there are no more objects available -> set finished as true
        if (len(self.object_list_not_placed) == 0):
            finished = True
        # computing the reward
        placed_parts_obs = self._list_to_array(self._get_list_placed_elements())
        rew, finished1, terminal_rew = self._get_reward(placed_parts_obs)
        # get the observation
        obs = self._get_obs()
        # assign additional reward either if we failed or if we finished upon success
        if (prior_exit):
            rew = -1.0
        else:
            #i.e. no prior exit
            # but if completed succesfully -> add additional reward of +1
            if (finished1):
                rew += 1.0
        return obs, rew, (finished or finished1 or prior_exit), {}