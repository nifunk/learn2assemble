'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import simple_box, target_shapes_sampled_points
from env import standart

from gym.utils import seeding
from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

import sobol_seq
import time
import copy

import os

from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import math

class StackBoxesEnv3D_multiple_obj_w_robot(Environment):
    '''
    Simplest environment which also has the robot in it.
    '''

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, robot_info=False, load_path=None):
        self.did_rendering = False
        self.robot_info = robot_info
        # tells whether the adjacency matrix also should be created,...
        self.add_connectivity = add_connectivity
        self.fully_connected = fully_connected
        # total_num_nodes describes the total number of nodes in the graph
        self.total_num_nodes = num_boxes
        self.num_boxes = self.total_num_nodes-3

        self.curr_time = 0
        self.reset_env = num_boxes-1

        self.viewer = None
        self.num_simulation_steps = 50
        #observation is; x,y, z,part placed, structure? -> 5 features in total
        # set ridgid observation limit
        obs_limit = 0.025 + 0.05*20
        self.obs_high = np.array([obs_limit,obs_limit,obs_limit,1,1],dtype=np.float32)
        self.obs_low = np.array([-obs_limit,-obs_limit,-obs_limit,-1,-1],dtype=np.float32)
        self.num_features = 5
        if self.add_connectivity:
            adj_array = np.ones(self.total_num_nodes)
            self.num_features += self.total_num_nodes
            self.obs_high = np.concatenate((self.obs_high,adj_array))
            self.obs_low = np.concatenate((self.obs_low,(-1*adj_array)))

        observation_high = self.obs_high
        observation_high_additional = self.obs_high
        observation_low = self.obs_low
        observation_low_additional = self.obs_low

        for i in range(self.total_num_nodes-1):
            observation_high = np.concatenate((observation_high,observation_high_additional))
            observation_low = np.concatenate((observation_low,observation_low_additional))

        self.observation_high = observation_high
        self.observation_low = observation_low

        self.seed(seed=seed)
        self.prev_rew = 0
        # Now define the actual environment in pybullet:
        self.SimEnv = standart.Standart(visualize=visualize)

        self.visualize = visualize
        self.object_list = []
        self.object_list_not_placed = []
        self.target_shape_list = []

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

        self.target_shapes = []

        if not(ensemble):
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0], self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquare(self.SimEnv.cam_list[0],self.rotation_operations[1], self.translation_operations[1], self.flip_lr_operations[1], width=env_grid_size,height=env_grid_size))
        else:
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))

        self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)

        self.ensemble = ensemble
        self.initial_elements_covered = 0

        self.populate_env()

        # set MDP properties
        action_space = Box(low=np.asarray([0,0,0]), high=np.asarray([self.total_num_nodes-1,self.total_num_nodes-1,4]), shape=(3,))
        if (robot_info==False):
            observation_space = Box(low=observation_low, high=observation_high, shape=np.shape(observation_low))
        else:
            pos_vel_low = -10*np.ones((2*9))
            pos_vel_high = 10*np.ones((2*9))
            observation_low = np.concatenate((observation_low, pos_vel_low))
            observation_high = np.concatenate((observation_high, pos_vel_high))
            observation_space = Box(low=observation_low, high=observation_high, shape=np.shape(observation_low))
        horizon = np.inf
        gamma = .8
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)
        self.SimEnv.add_lab_setup()

    def get_random_target_shape_multiple(self):
        self.target_shape_mask = []
        self.target_shape_list = []
        for i in range(len(self.target_shapes)):
            target_shape_mask_tmp, target_shape_list_tmp = self.target_shapes[i].get_random_target_shape()
            self.target_shape_mask.append(target_shape_mask_tmp)
            self.target_shape_list += target_shape_list_tmp
        return self.target_shape_mask, self.target_shape_list

    def init_random_target_shape_multiple(self,idx_list,point_list):
        first = 0
        self.target_shape_mask = []
        self.target_shape_list = []
        for i in range(len(self.target_shapes)):
            target_shape_mask_tmp, target_shape_list_tmp = self.target_shapes[i].get_random_target_shape(point_list[first:first+idx_list[i]])
            first += idx_list[i]
            self.target_shape_mask.append(target_shape_mask_tmp)
            self.target_shape_list += target_shape_list_tmp
        return self.target_shape_mask, self.target_shape_list

    def visualize_target_shape_multiple(self):
        for i in range(len(self.target_shapes)):
            self.target_shapes[i].visualize_shape(self.SimEnv.physicsClient)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.did_rendering = False

        if (self.visualize):
            self.visualize_target_shape_multiple()
            time.sleep(0.5)

        self.curr_time += 1
        # find out number of objects placed currently in the environment, before adding the new one:
        curr_num_targets = len(self.target_shape_list)
        curr_num_obj = len(self.object_list)
        curr_num_objects_not_placed = len(self.object_list_not_placed)
        # decompose the action
        block_to_be_placed = np.clip(u[0],curr_num_targets+curr_num_obj, curr_num_targets+curr_num_obj+curr_num_objects_not_placed-1)-(curr_num_targets+curr_num_obj)
        reference_block = np.clip(u[1],curr_num_targets, curr_num_targets+curr_num_obj-1)-(curr_num_targets)
        # retrieve the block that is to be moved
        obj_to_be_moved = self.object_list_not_placed.pop(block_to_be_placed)
        pos, ori = obj_to_be_moved.get_pos_orient()
        # determine the placing action. the offset list contains the goto points that are reached after one another
        # to ensure the proper placing of the block
        p_offsets = []
        dist = 0.055
        if (u[2]==0):
            idx = 4
            # on top:
            p_offsets.append([0,0,0.25])
            p_offsets.append([0,0,0.075])
            p_offsets.append([0,0,0.055])
            p_offsets.append([0,0,0.05+0.005])
        elif (u[2]==1):
            idx = 1
            p_offsets.append([-dist,0,0.25])
            p_offsets.append([-dist,0,0.075-0.05])
            p_offsets.append([-dist,0,0.055-0.05+0.005])
            p_offsets.append([-dist,0,0.05-0.05+0.005])
        elif (u[2] == 2):
            idx = 0
            p_offsets.append([dist,0,0.25])
            p_offsets.append([dist,0,0.075-0.05])
            p_offsets.append([dist,0,0.055-0.05+0.005])
            p_offsets.append([dist,0,0.05-0.05+0.005])
        elif (u[2] ==3):
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


        # attempt to grasp the part
        successfull = self.SimEnv.grasp_part(pos,[-1, -0., 0., 0.],obj_to_be_moved)

        self.object_list.append(obj_to_be_moved)
        # store position of all currently placed parts
        self.curr_placed_parts_pos = []
        for i in range(len(self.object_list)-1):
            self.curr_placed_parts_pos.append(self.object_list[i].get_pos_orient()[0])

        lower_obj = self.object_list[reference_block]
        pos, ori = lower_obj.get_pos_orient()
        if (successfull):
            # now attempt to actually place the part
            successfull = self.SimEnv.place_part(pos,p_offsets,[1, -0., 0., 0.],lower_obj,obj_to_be_moved, self.object_list)

        prior_exit = False
        # after placing the object - check if all other placed parts are also still in place
        for i in range(len(self.object_list)-1):
            diff = np.abs(np.asarray(self.object_list[i].get_pos_orient()[0])-np.asarray(self.curr_placed_parts_pos[i]))
            diff = np.sqrt(np.sum(np.multiply(diff,diff)))
            if (diff>0.035):
                prior_exit=True

        if (not(successfull)):
            prior_exit = True

        finished = False
        if (len(self.object_list_not_placed)==0):
            finished = True
        # compute the reward
        placed_parts_obs = self._list_to_array(self._get_list_placed_elements())
        rew, finished1, terminal_rew = self._get_reward(placed_parts_obs)
        # get the observation
        obs = self._get_obs()
        # if any invalid action has been conducted - add additional penalty
        if (prior_exit):
            rew = -1.0
        return obs, rew, (finished or finished1 or prior_exit), {}

    def reset(self, state=None):
        # compared to other implementations - first reset the robot
        self.SimEnv.reset_robot()
        self.did_rendering = False
        target_elem_list = []
        placed_elem_list = []
        to_be_placed_list = []

        if not(state is None):
            # do potential shortcut if env already at right state
            prev_obs = self._get_obs()
            if (np.all(state==prev_obs)):
                # do this once to init previous reward,...
                rew, finished1, terminal_rew = self._get_reward(None)
                return prev_obs

            target_elem, placed_elem, to_be_placed, num_actions = self._decode_observation(state)
            total_elements = len(target_elem) + len(placed_elem) + len(to_be_placed)
            if (self.robot_info):
                self.SimEnv.reset_robot(state[-18:-9], state[-9:])
                state = self.preprocess_decode(state)
            len_individ_obs = int(np.shape(state)[0]/total_elements)
            obs_reshaped = np.asarray(state).reshape(-1,len_individ_obs)

            total_counter = 0
            for i in range(len(target_elem)):
                target_elem_list.append([obs_reshaped[total_counter,0],obs_reshaped[total_counter,1],obs_reshaped[total_counter,2]])
                total_counter+=1
            for i in range(len(placed_elem)):
                placed_elem_list.append([obs_reshaped[total_counter,0],obs_reshaped[total_counter,1],obs_reshaped[total_counter,2]])
                total_counter+=1
            for i in range(len(to_be_placed)):
                to_be_placed_list.append([obs_reshaped[total_counter,0],obs_reshaped[total_counter,1],obs_reshaped[total_counter,2]])
                total_counter+=1

            self.curr_time = len(placed_elem_list)-1

        else:
            self.curr_time = 0


        if (self.visualize):
            time.sleep(1)
        self.prev_rew = 0
        self.prev_iou = 0
        self.empty_env()
        if state is None:
            self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        else:
            # have to decompose the list of the target elements to initialize the environment correctly
            interval_list = []
            copied_list = copy.deepcopy(target_elem_list)
            for i in range(len(self.target_shapes)):
                local_copy = copy.deepcopy(copied_list)
                local_copy = np.transpose(np.matmul(np.linalg.inv(self.rotation_operations[i]),
                                                                    np.add(np.transpose(local_copy),
                                                                           -self.translation_operations[i].reshape(3, -1))))
                local_copy = list(local_copy)
                first_element = local_copy[0][1]
                copied_list.pop(0)
                for j in range(len(local_copy)-1):
                    if (np.abs(local_copy[j+1][1]-(first_element))<0.0001):
                        copied_list.pop(0)
                        j+=1
                    else:
                        interval_list.append(j+1)
                        break
                    if (j == len(local_copy) - 1):
                        interval_list.append(j+1)
            # initialize the target shape based on observation
            self.target_shape_mask, self.target_shape_list = self.init_random_target_shape_multiple(interval_list,target_elem_list)

        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)
        self.populate_env(default_placed=placed_elem_list, default_unplaced=to_be_placed_list)

        while(True):
            self.did_rendering = False
            self.total_elements = np.sum(self.target_shape_mask)
            self.initial_elements_covered = np.sum(np.multiply(self._get_depth(),self.target_shape_mask))
            unified_area = np.sum(np.clip(self._get_depth()+self.target_shape_mask,0,1))
            intersecton = np.sum(np.multiply(self._get_depth(),self.target_shape_mask))
            self.prev_iou = intersecton / unified_area
            # hand designed conditions on when sampled target shape is valid
            if (self.total_elements<50 or (self.initial_elements_covered/self.total_elements)>0.6):
                # if invalid -> sample anew:
                self.empty_env()
                self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
                self.num_boxes = self.total_num_nodes - len(self.target_shape_list)
                self.populate_env()
            else:
                break

        obs = self._get_obs()
        # call once get reward to init things,..
        rew, finished1, terminal_rew = self._get_reward(None)

        return obs

    def _list_to_array(self,list):
        # converts list to array
        obs_array = list[0]
        for i in range(len(list)-1):
            obs_array = np.concatenate((obs_array,list[i+1]))
        return obs_array

    def _get_list_placed_elements(self):
        # returns list of the placed elements
        obs_list = []
        obs_list.append(self.object_list[0].get_pos_orient()[0])
        for i in range(len(self.object_list)-1):
            obs_list.append(self.object_list[i+1].get_pos_orient()[0])
        return obs_list

    def _get_list_unplaced_elements(self):
        # returns list of unplaced elements
        obs_list = []
        for i in range(len(self.object_list_not_placed)):
            obs_list.append(self.object_list_not_placed[i].get_pos_orient()[0])
        return obs_list

    def _get_depth(self,filter=None):
        # function returning depth images of the environment which can be filtered based on provided filtering intervals
        if not(self.did_rendering):
            self.SimEnv.render()
            self.did_rendering = True
            self.raw_image = []
            for i in range(len(self.target_shapes)):
                self.raw_image.append(copy.deepcopy(self.SimEnv.get_depth(cam_id=i)))
        self.image = []
        for i in range(len(self.target_shapes)):
            if (filter is None):
                self.image.append(self._do_filtering(copy.deepcopy(self.raw_image[i]),filter=[0.21,0.25]))
            else:
                self.image.append(self._do_filtering(copy.deepcopy(self.raw_image[i]), filter=filter))
        return self.image

    def _do_filtering(self,image,filter):
        # function implements filtering on array based on a specified interval
        image[image < filter[0]] = 0
        image[image > filter[1]] = 0
        image[image != 0] = 1
        return image

    def _get_obs(self):
        # returns the observation
        target_elements = self.target_shape_list
        real_elements = self._get_list_placed_elements()
        unplaced_elements = self._get_list_unplaced_elements()
        obs = self._encode_observation(target_elements,real_elements,unplaced_elements)

        return obs

    def _encode_observation(self,target_elements,real_elements,unplaced_elements):
        # function to encode the observation
        # note here now depending on the setup, also the information of the robot is being added
        add_array = np.asarray([])
        if self.add_connectivity:
            add_array = np.ones(self.total_num_nodes)
            add_array_existing_blocks = copy.deepcopy(add_array)
            if (not self.fully_connected):
                start_unplaced_elem = len(target_elements)+len(real_elements)
                add_array[start_unplaced_elem:] = 0

        total_counter = 0
        obs_array = np.concatenate((np.asarray(target_elements[0]),np.asarray([1,1]),add_array_existing_blocks))
        total_counter += 1
        for i in range(len(target_elements)-1):
            obs_array1 =np.concatenate((np.asarray(target_elements[i+1]),np.asarray([1,1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1

        for i in range(len(real_elements)):
            obs_array1 =np.concatenate((np.asarray(real_elements[i]),np.asarray([1,-1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1

        for i in range(len(unplaced_elements)):
            if (not self.fully_connected):
                new_arr = copy.deepcopy(add_array)
                new_arr[int(total_counter)] = 1.0
                obs_array1 =np.concatenate((np.asarray(unplaced_elements[i]),np.asarray([-1,-1]),new_arr))
            else:
                obs_array1 =np.concatenate((np.asarray(unplaced_elements[i]),np.asarray([-1,-1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1

        if (self.robot_info):
            robot_pos, robot_vel = self.SimEnv.get_robot_info()
            obs_array = np.concatenate((obs_array, np.asarray(robot_pos)))
            obs_array = np.concatenate((obs_array, np.asarray(robot_vel)))

        return obs_array

    def _decode_observation(self,state):
        # decoding the observation into target elements, elements placed, elements to be placed and range of action space
        dimensions = self.num_features
        new_state = copy.deepcopy(state)
        if (self.robot_info):
            new_state = self.preprocess_decode(new_state)
        new_state = np.reshape(new_state,(-1,dimensions))
        target_elements = np.where(np.logical_and(new_state[:,3]==1,new_state[:,4]==1)==1)[0]
        elements_placed = np.where(np.logical_and(new_state[:,3]==1,new_state[:,4]==-1)==1)[0]
        elements_to_be_placed = np.where(np.logical_and(new_state[:,3]==-1,new_state[:,4]==-1)==1)[0]
        return target_elements, elements_placed, elements_to_be_placed, np.arange(5)

    def _decode_observation_multidim(self,state):
        # decoding the observation into target elements, elements placed, elements to be placed and range of action space
        # also returns some arrays that indicate at which indices there are target elements and placed / unplaced blocks
        dimensions = self.num_features
        new_state = copy.deepcopy(state)
        if (len(np.shape(new_state)) == 1):
            new_state = new_state.reshape(1,-1)
        if (self.robot_info):
            new_state = self.preprocess_decode_multidim(new_state)
        new_state = np.reshape(new_state,(np.shape(new_state)[0],-1,dimensions))

        target_arr = (np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==1))
        target_elements = (np.argwhere(np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==1)==1))

        elements_placed_arr = np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==-1)
        elements_placed = np.argwhere(np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==-1)==1)

        elements_to_be_placed = np.argwhere(np.logical_and(new_state[:,:,3]==-1,new_state[:,:,4]==-1)==1)
        elements_to_be_placed_arr = np.logical_and(new_state[:,:,3]==-1,new_state[:,:,4]==-1)

        return target_elements, elements_placed, elements_to_be_placed, np.arange(5), target_arr, elements_placed_arr, elements_to_be_placed_arr #also add that there are 3 actions in total

    def _get_env_info(self):
        # get the information of the environment - needed for logging mostly
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
            return_rew = 0.0
        else:
            return_rew = 1.0

        finished = False
        if (len(self.object_list_not_placed)==0):
            finished = True

        return return_rew, curr_iou, (curr_rew_complete > 0.75), finished

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
        # discrete reward signal - if filling increased give +1 - otherwise 0
        if (return_rew<=0):
            return_rew = 0.0
        else:
            return_rew = 1.0

        return return_rew, (curr_rew_complete>0.75), curr_rew1

    def render(self, mode='human'):
        # functionality required by the env
        return 0

    def close(self):
        # functionality required by the env
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def populate_env(self,default_placed=[], default_unplaced=[]):
        if not(len(self.object_list)==0):
            self.empty_env()

        if (default_placed==[] and default_unplaced==[]):
            # if no prespecifications -> place first block randomly
            init_idx = np.random.choice(len(self.target_shapes))
            self.object_list.append(simple_box.Simple_box([self.translation_operations[init_idx][0], self.translation_operations[init_idx][1], 0], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            # calculate number of missing samples
            missing_samples = self.num_boxes - len(self.object_list)

            # use sobol sequences to add additional parts evenly
            divider_samples = 6
            offset_y = -0.2
            num_rounds = int(math.ceil(missing_samples/divider_samples))
            total_samples = missing_samples
            # place parts carefully as they should not fall off the table in this real environment
            for j in range(num_rounds):
                if (total_samples>=divider_samples):
                    current_samples = divider_samples
                else:
                    current_samples = total_samples
                total_samples -= current_samples
                coord_x = sobol_seq.i4_sobol_generate(1,current_samples)
                coord_x = (2*coord_x - 1)*0.75*self.observation_high[1]/2 - 0.15*self.observation_high[1]/2
                for i in range(current_samples):
                    self.object_list_not_placed.append( \
                    simple_box.Simple_box(
                        [coord_x[i,0],offset_y, 0.0],
                        [0, 0, 0, 1], self.SimEnv.physicsClient))
                offset_y += -0.125
        else:
            # otherwise if all positions are prespecified -> place them accordingly
            self.object_list.append(simple_box.Simple_box([default_placed[0][0], default_placed[0][1], default_placed[0][2]-0.05 / 2], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            for i in range(len(default_placed)-1):
                self.object_list.append(
                    simple_box.Simple_box([default_placed[i+1][0], default_placed[i+1][1], default_placed[i+1][2]-0.05 / 2], [0, 0, 0, 1],
                                          self.SimEnv.physicsClient))
            for i in range(len(default_unplaced)):
                self.object_list_not_placed.append(
                    simple_box.Simple_box([default_unplaced[i][0], default_unplaced[i][1], default_unplaced[i][2] - 0.05 / 2],
                                          [0, 0, 0, 1], self.SimEnv.physicsClient))

    def empty_env(self):
        # remove all objects from the environment
        for i in range(len(self.object_list)):
            self.object_list[i].remove()
        self.object_list.clear()
        for i in range(len(self.object_list_not_placed)):
            self.object_list_not_placed[i].remove()
        self.object_list_not_placed.clear()