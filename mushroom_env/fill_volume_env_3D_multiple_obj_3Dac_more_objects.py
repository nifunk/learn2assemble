'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import parts_interactions, simple_box, target_shapes_sampled_points
from parts import create_complex_objects

from env import standart

from gym.utils import seeding

from mushroom_rl.environments import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

import sobol_seq
import time
import copy

from scipy.spatial.transform import Rotation as R

class StackBoxesEnv3D_multiple_obj(Environment):
    '''
    Environment is the simplest one to deal with the more complex objects. This implies i) more complex objects are used
    which are a composition of the promitive boxes
    '''

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, load_path=None, num_actions=5, disc_factor=None,fill_threshold=0.975):
        self.fill_threshold = fill_threshold
        self.num_actions = num_actions
        self.obj_creator = create_complex_objects.ComplexObjects()
        self.did_rendering = False
        # tells whether the adjacency matrix also should be created,...
        self.add_connectivity = add_connectivity
        self.fully_connected = fully_connected
        # total_num_nodes is the total number of nodes in the graph. As the RL algorithms have fixed size observations
        # the total number of nodes is also fixed
        self.total_num_nodes = num_boxes
        self.num_boxes = self.total_num_nodes-3
        self.curr_time = 0
        self.reset_env = num_boxes-1

        self.viewer = None
        self.num_simulation_steps = 50

        # observation is; x,y,z,part placed, structure? -> coincides to 5 features
        # set ridgid observation limit
        obs_limit = 0.025 + 0.05*20
        self.obs_high = np.array([obs_limit,obs_limit,obs_limit,1,1],dtype=np.float32)
        self.obs_low = np.array([-obs_limit,-obs_limit,-obs_limit,-1,-1],dtype=np.float32)
        self.num_features = 5
        self.num_features_orig = 3 # number of features only related with pos (x,y,z)

        if self.add_connectivity:
            adj_array = np.ones(self.total_num_nodes) # times two as we also have target blocks,...
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
        else:
            self.target_shapes.append(target_shapes_sampled_points.TargetsBlockSquareUp5(self.SimEnv.cam_list[0],self.rotation_operations[0], self.translation_operations[0], self.flip_lr_operations[0], width=env_grid_size,height=env_grid_size))
        self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)

        self.ensemble = ensemble
        self.initial_elements_covered = 0
        self.additional_blocks_initial = 0
        self.standard_image = copy.deepcopy(self._get_depth(filter=[0.0, 0.226]))
        # self.SimEnv.show_depth(filter=[0.0, 0.226])
        self.populate_env()

        # MDP properties
        action_space = Box(low=np.asarray([0,0,0]), high=np.asarray([self.total_num_nodes-1,self.total_num_nodes-1,self.num_actions-1]), shape=(3,))
        observation_space = Box(low=observation_low, high=observation_high, shape=np.shape(observation_low))
        horizon = np.inf  # the gym time limit is used.
        if (disc_factor is None):
            gamma = .8
        else:
            gamma = disc_factor
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

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
        # find out number of objects before adding the new one:
        curr_num_targets = len(self.target_shape_list)
        curr_num_obj = len(self.object_list)
        curr_num_objects_not_placed = len(self.object_list_not_placed)
        # add the new part:
        self.object_list.append(simple_box.Simple_box([0.0, 0.0, 0], [0, 0, 0, 1],self.SimEnv.physicsClient))
        self.object_list[-1].apply_velocity([0, 0, 0])

        # decode the action (block to be placed - reference block - relative action)
        block_to_be_placed = np.clip(u[0],curr_num_targets+curr_num_obj, curr_num_targets+curr_num_obj+curr_num_objects_not_placed-1)-(curr_num_targets+curr_num_obj)
        reference_block = np.clip(u[1],curr_num_targets, curr_num_targets+curr_num_obj-1)-(curr_num_targets)
        if (u[2]==0):
            idx = 4
        elif (u[2]==1):
            idx = 1
        elif (u[2] == 2):
            idx = 0
        elif (u[2] ==3):
            idx = 3
        else:
            idx = 2

        lower_obj = self.object_list[reference_block]
        upper_obj = self.object_list[-1]
        # do actual placing of the part
        success = parts_interactions.stack_complex_parts_primitively(lower_obj,upper_obj,idx,self.object_list,self.object_list_not_placed, block_to_be_placed, self.SimEnv.physicsClient)

        # get how many objects are remaining -> as we now have objects made out of multiple blocks, this is more complex
        missing_samples = self.obj_creator.get_num_remaining_blocks(self.object_list_not_placed)

        if not success:
            print ("ERROR IN THE ENVIRONMENT")

        # check stability of the created structure by tracking the velocities of all parts
        acc_vel_lin = 0.0
        acc_vel_ang = 0.0
        for j in range(self.num_simulation_steps):
            self.SimEnv.step()
            for i in range(len(self.object_list)):
                curr_lin, curr_ang = self.object_list[i].get_vel()
                acc_vel_lin += np.sum(np.abs(curr_lin))
                acc_vel_ang += np.sum(np.abs(curr_ang))

        prior_exit = False
        # if this threshold is exceeded -> we performed an invalid move
        if (acc_vel_lin>1.0):
            prior_exit = True

        finished = False
        # terminate if there are no more blocks available
        if (missing_samples==0):
            finished = True

        placed_parts_obs = self._list_to_array(self._get_list_placed_elements())
        # compute the reward
        rew, finished1, terminal_rew = self._get_reward(placed_parts_obs)

        # get the observation
        obs = self._get_obs()

        if (prior_exit):
            prior_exit = True
            rew = -1.0
        # if finished successfully - assign additional reward of 1
        elif (finished1):
            rew += 1

        return obs, rew, (finished or finished1 or prior_exit), {}

    def reset(self, state=None):
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
            # otherwise decompose observation and fill env as specified by observation
            target_elem, placed_elem, to_be_placed, num_actions = self._decode_observation(state)
            total_elements = len(target_elem) + len(placed_elem) + len(to_be_placed)
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
        self.previous_unified_area = 0
        self.previous_intersection_area = 0
        self.empty_env()
        if state is None:
            self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()
        else:
            # have to decompose the list to initialize the structure that is to be built correctly
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
                        copied_list.pop(0)  # remove first element always,...
                        j+=1
                    else:
                        interval_list.append(j+1)
                        break
                    if (j == len(local_copy) - 1):
                        interval_list.append(j+1)

            self.target_shape_mask, self.target_shape_list = self.init_random_target_shape_multiple(interval_list,target_elem_list)

        self.num_boxes = self.total_num_nodes - len(self.target_shape_list)
        self.populate_env(default_placed=placed_elem_list, default_unplaced=to_be_placed_list)

        while(True):
            self.did_rendering = False
            self.total_elements = np.sum(self.target_shape_mask)
            self.initial_elements_covered = np.sum(np.multiply(self._get_depth(),self.target_shape_mask))
            self.additional_blocks_initial = np.sum(np.clip((np.subtract(np.subtract(self._get_depth(filter=[0.0,0.226]),self.target_shape_mask),self.standard_image)),0,1))

            unified_area = np.sum(np.clip(self._get_depth()+self.target_shape_mask,0,1))
            intersecton = np.sum(np.multiply(self._get_depth(),self.target_shape_mask))
            self.prev_iou = intersecton / unified_area
            self.previous_unified_area = unified_area
            self.previous_intersection_area = intersecton
            # hand designed features that should ensure that target shape that is to be built is sufficiently large and
            # also not already covered completely by the initially placed red block
            if (self.total_elements<50 or (self.initial_elements_covered/self.total_elements)>0.95):
                # if one of the conditions is violated -> sample anew
                self.empty_env()
                self.target_shape_mask, self.target_shape_list = self.get_random_target_shape_multiple()#target_shapes.get_random_target_shape()
                self.num_boxes = self.total_num_nodes - len(self.target_shape_list)
                self.populate_env()
            else:
                break

        obs = self._get_obs()
        # call once get reward to init things,..
        rew, finished1, terminal_rew = self._get_reward(None)

        return obs

    def _list_to_array(self,list):
        # converts list into array
        obs_array = list[0]
        for i in range(len(list)-1):
            obs_array = np.concatenate((obs_array,list[i+1]))
        return obs_array

    def _list_to_array_idx(self,list, idx):
        # returns the positions of the currently placed elements as an array
        obs_array = list[0][:idx]
        for i in range(len(list)-1):
            obs_array = np.concatenate((obs_array,list[i+1][:idx]))
        return obs_array

    def _get_list_placed_elements(self):
        # returns the positions of the currently placed elements as a list
        obs_list = []
        obs_list.append(self.object_list[0].get_pos_orient()[0])
        for i in range(len(self.object_list)-1):
            obs_list.append(self.object_list[i+1].get_pos_orient()[0])
        return obs_list

    def _get_depth(self,filter=None):
        # get the depth information by rendering all the cameras that are placed in the scene
        if not(self.did_rendering):
            self.SimEnv.render()
            self.did_rendering = True
            self.raw_image = []
            for i in range(len(self.target_shapes)):
                self.raw_image.append(copy.deepcopy(self.SimEnv.get_depth(cam_id=i)))
        self.image = []
        for i in range(len(self.target_shapes)):
            if (filter is None):
                self.image.append(self._do_filtering(copy.deepcopy(self.raw_image[i]),filter=[0.224,0.226]))
            else:
                self.image.append(self._do_filtering(copy.deepcopy(self.raw_image[i]), filter=filter))
        return self.image

    def _do_filtering(self,image,filter):
        # easy function to filter the depth image based on a provided interval
        image[image < filter[0]] = 0
        image[image > filter[1]] = 0
        image[image != 0] = 1
        return image

    def _get_obs(self):
        # returns the observation
        target_elements = self.target_shape_list
        real_elements = self._get_list_placed_elements()
        unplaced_elements = self.object_list_not_placed
        obs = self._encode_observation(target_elements,real_elements,unplaced_elements)
        return obs

    def _encode_observation(self,target_elements,real_elements,unplaced_elements):
        # function that creates the observation by encoding the information that is stored in the environment
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
        # first add target elements and real elements as previously with the simpler blocks
        for i in range(len(target_elements)-1):
            obs_array1 =np.concatenate((np.asarray(target_elements[i+1][:self.num_features_orig]),np.asarray([1,1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1
        for i in range(len(real_elements)):
            obs_array1 =np.concatenate((np.asarray(real_elements[i][:self.num_features_orig]),np.asarray([1,-1]),add_array_existing_blocks))
            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1
        # as the unplaced blocks might now also be a composition of the simple blocks, in the partially connected setting
        # the connectivity information has to be computed appropriately,...
        for i in range(len(unplaced_elements)):
            if (not self.fully_connected):
                # the connectivity is provided in the list of the unplaced elements after their position (x,y,z)
                # which parts are connected is also given through a relative indexing
                connected_elements = np.asarray(unplaced_elements[i][3:])
                # this operation adjusts the relative indexing such that it becomes a global one, exploiting the info of
                # how many blocks have been placed already
                connected_elements = (connected_elements + total_counter).astype(int)
                new_arr = copy.deepcopy(add_array)
                new_arr[connected_elements] = 1.0 # -> set also those entries given by the indexing to 1 -> there is connection
                obs_array1 =np.concatenate((np.asarray(unplaced_elements[i][:self.num_features_orig]),np.asarray([-1,-1]),new_arr))
            else:
                obs_array1 =np.concatenate((np.asarray(unplaced_elements[i][:self.num_features_orig]),np.asarray([-1,-1]),add_array_existing_blocks))

            obs_array = np.concatenate((obs_array,obs_array1))
            total_counter += 1

        return obs_array

    def _decode_observation(self,state):
        # decode observation into target elements, placed elements, elements to be placed and num of actions
        dimensions = self.num_features
        new_state = copy.deepcopy(state)
        new_state = np.reshape(new_state,(-1,dimensions))
        target_elements = np.where(np.logical_and(new_state[:,3]==1,new_state[:,4]==1)==1)[0]
        elements_placed = np.where(np.logical_and(new_state[:,3]==1,new_state[:,4]==-1)==1)[0]
        elements_to_be_placed = np.where(np.logical_and(new_state[:,3]==-1,new_state[:,4]==-1)==1)[0]
        return target_elements, elements_placed, elements_to_be_placed, np.arange(self.num_actions)

    def _decode_observation_multidim(self,state):
        # decode observation into target elements, placed elements, elements to be placed and num of actions AND
        # arrays that allow to filter everything more easily by setting 1 at the corresponding indices
        dimensions = self.num_features
        new_state = copy.deepcopy(state)
        if (len(np.shape(new_state)) == 1):
            new_state = new_state.reshape(1,-1)
        new_state = np.reshape(new_state,(np.shape(new_state)[0],-1,dimensions))

        target_arr = (np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==1))
        target_elements = (np.argwhere(np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==1)==1))

        elements_placed_arr = np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==-1)
        elements_placed = np.argwhere(np.logical_and(new_state[:,:,3]==1,new_state[:,:,4]==-1)==1)

        elements_to_be_placed = np.argwhere(np.logical_and(new_state[:,:,3]==-1,new_state[:,:,4]==-1)==1)
        elements_to_be_placed_arr = np.logical_and(new_state[:,:,3]==-1,new_state[:,:,4]==-1)

        return target_elements, elements_placed, elements_to_be_placed, np.arange(self.num_actions), target_arr, elements_placed_arr, elements_to_be_placed_arr

    def _get_env_info(self):
        # get info of the environment -> this is mainly needed for logging purposes
        elem_covered = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_rew1 = ((elem_covered-self.initial_elements_covered)/(self.total_elements-self.initial_elements_covered))
        #Calculate everything with holistic view again:
        elem_covered_complete = np.sum(np.multiply(self._get_depth(filter=[0.0,0.226]), self.target_shape_mask))
        curr_rew_complete = ((elem_covered_complete-self.initial_elements_covered)/(self.total_elements-self.initial_elements_covered))

        unified_area = np.sum(np.clip(self._get_depth() + self.target_shape_mask, 0, 1))
        intersecton = np.sum(np.multiply(self._get_depth(filter=[0.0,0.226]), self.target_shape_mask))
        curr_iou = intersecton / unified_area

        finished = False
        missing_samples = self.obj_creator.get_num_remaining_blocks(self.object_list_not_placed)
        if (missing_samples==0):
            finished = True
        return curr_rew_complete, curr_iou, (curr_rew_complete>self.fill_threshold), finished

    def _get_reward(self,obs):
        # compute reward
        elem_covered = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_rew1 = ((elem_covered-self.initial_elements_covered)/(self.total_elements-self.initial_elements_covered))
        #Calculate everything with holistic view again:
        elem_covered_complete = np.sum(np.multiply(self._get_depth(filter=[0.0,0.226]), self.target_shape_mask))
        curr_rew_complete = ((elem_covered_complete-self.initial_elements_covered)/(self.total_elements-self.initial_elements_covered))

        return_rew = curr_rew1 - self.prev_rew
        self.prev_rew = curr_rew1
        unified_area = np.sum(np.clip(self._get_depth() + self.target_shape_mask, 0, 1))
        intersecton = np.sum(np.multiply(self._get_depth(), self.target_shape_mask))
        curr_iou = intersecton / unified_area
        self.prev_iou = curr_iou
        self.previous_unified_area = unified_area
        self.previous_intersection_area = intersecton
        # here we are in continous case -> reward proportional to how much the filling has improved
        # Note: as filling is also considering whether parts are placed at the right depth, the relative change in filling
        # might also be negative -> however, we clip the reward to be purely positive
        if (return_rew<=0):
            return_rew = 0.0
        else:
            return_rew = return_rew

        return 3*return_rew, (curr_rew_complete>self.fill_threshold), curr_rew1

    def render(self, mode='human'):
        # function only needed to be in line with functionality required for valid environment
        return 0

    def close(self):
        # function only needed to be in line with functionality required for valid environment
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def populate_env(self,default_placed=[], default_unplaced=[]):
        # if env not already empty - do it
        if not(len(self.object_list)==0):
            self.empty_env()

        if (default_placed==[] and default_unplaced==[]):
            # if environment is empty - sample a set of blocks
            list_of_objects = self.obj_creator.sample_blockset(self.num_boxes)

            # now only place one initial box in the center at random side (if multiple sides are given)
            init_idx = np.random.choice(len(self.target_shapes))
            self.object_list.append(simple_box.Simple_box([self.translation_operations[init_idx][0], self.translation_operations[init_idx][1], 0], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            # NOW POTENTIALLY REPOSITION THIS WHOLE THINGY,...
            self.list_all_obj = []
            dimensions = self.object_list[0].dimensions
            for i in range(len(list_of_objects)):
                for j in range(len(list_of_objects[i])):
                    # add scale to this list of objects that has been created (as intially all is again relative)
                    list_of_objects[i][j][:3] = (np.multiply(np.asarray(list_of_objects[i][j][:3]),2*np.asarray(dimensions))).tolist()
                    self.list_all_obj.append(list_of_objects[i][j])
            # NOW POSITION FIRST OBJECT CORRECTLY, i.e. place entire object
            parts_interactions.stack_complex_parts_primitively_absolute(self.object_list[0],None,self.list_all_obj,self.object_list,self.SimEnv.physicsClient)
            # retrieve remaining number of objects
            missing_samples = self.obj_creator.get_num_remaining_blocks(self.list_all_obj)
            # sample positions for the remaining blocks
            coord_x = sobol_seq.i4_sobol_generate(1,missing_samples)
            coord_x = (2*coord_x - 1)*self.observation_high[1]
            # place additional blocks
            self.obj_creator.place_unplaced(self.list_all_obj,self.object_list_not_placed,coord_x,[0,-0.05,0.0])
            unplaced_as_arr = self._list_to_array_idx(self.object_list_not_placed,3).reshape(-1,3)
            # check x,y,z coords, i.e. such that they are in bound
            check_x_coord = np.sum((np.logical_and(unplaced_as_arr[:,0]>self.observation_low[0], unplaced_as_arr[:,0]<self.observation_high[0]))-1)
            check_y_coord = np.sum((np.logical_and(unplaced_as_arr[:,1]>self.observation_low[1], unplaced_as_arr[:,1]<self.observation_high[1]))-1)
            check_z_coord = np.sum((np.logical_and(unplaced_as_arr[:,2]>self.observation_low[2], unplaced_as_arr[:,2]<self.observation_high[2]))-1)
            if ((check_x_coord+check_y_coord+check_z_coord)!=0):
                print ("CAUTION: ONE OF THE PARTS IS PLACED OUT OF BOUNDS,....")
        else:
            self.object_list.append(simple_box.Simple_box([default_placed[0][0], default_placed[0][1], default_placed[0][2]-0.05 / 2], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            # first place all blocks that are anyways already placed
            for i in range(len(default_placed)-1):
                self.object_list.append(
                    simple_box.Simple_box([default_placed[i+1][0], default_placed[i+1][1], default_placed[i+1][2]-0.05 / 2], [0, 0, 0, 1],
                                          self.SimEnv.physicsClient))
            # for the unplaced -> before placing them -> have to check whether they are close to each other, i.e. whether
            # they actually belong together and form something bigger
            unplaced_elements_list = self.obj_creator.decompose_unplaced(copy.deepcopy(default_unplaced), 2*np.asarray(self.object_list[0].dimensions))
            for i in range(len(default_unplaced)):
                self.object_list_not_placed.append([*unplaced_elements_list[i]])

        self.yolo_list = []

    def empty_env(self):
        # remove all objects from the env
        for i in range(len(self.object_list)):
            self.object_list[i].remove()
        for i in range(len(self.yolo_list)):
            self.yolo_list[i].remove()
        self.object_list.clear()
        self.object_list_not_placed.clear()
        self.yolo_list.clear()
