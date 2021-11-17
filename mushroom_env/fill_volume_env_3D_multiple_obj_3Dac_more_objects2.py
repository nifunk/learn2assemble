'''
This file is intended to provide a "gym"-like interface to the stacking boxes scenario,...
'''
import sys
sys.path.append("../")
from parts import parts_interactions, simple_box, target_shapes_sampled_points

from mushroom_rl.utils.spaces import *

import sobol_seq
import time
import copy

from scipy.spatial.transform import Rotation as R
from .fill_volume_env_3D_multiple_obj_3Dac_more_objects import StackBoxesEnv3D_multiple_obj

from parts import create_complex_objects

class StackBoxesEnv3D_multiple_obj(StackBoxesEnv3D_multiple_obj):
    '''
    This environment inherits everything from fill_volume_env_3D_multiple_obj_3Dac_more_objects_new.py and only adds a
    second side to it
    '''

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, num_boxes, visualize=True, seed=None, add_connectivity=False, ensemble=False, env_grid_size=3, fully_connected=True, load_path=None, num_actions=5, disc_factor=None,fill_threshold=0.975):
        self.check_stability = True
        super().__init__(num_boxes, visualize, seed, add_connectivity, ensemble, env_grid_size, fully_connected,
                         load_path,num_actions,disc_factor,fill_threshold)

        self.obj_creator = create_complex_objects.ComplexObjects3()
        # remove all cameras:
        self.SimEnv.remove_all_cameras()
        # define multiple cameras now:
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
        # again so far only 5 actions available top / left / right / front / bottom
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
        return idx

    def step(self, u):
        self.did_rendering = False

        #retrieve list of current positions of all blocks:
        curr_pos_list = []
        for i in range(len(self.object_list)):
            curr_pos_list.append(self.object_list[i].get_pos_orient()[0])
        curr_pos_list = np.asarray(curr_pos_list).reshape(-1, 3)

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
        # decode relative action (i.e. block to be placed, reference block,...)
        block_to_be_placed = np.clip(u[0],curr_num_targets+curr_num_obj, curr_num_targets+curr_num_obj+curr_num_objects_not_placed-1)-(curr_num_targets+curr_num_obj)
        reference_block = np.clip(u[1],curr_num_targets, curr_num_targets+curr_num_obj-1)-(curr_num_targets)
        # this contains all of the indices of potentially other blocks that are connected with the one to be placed
        invalid_idx = np.asarray(self.object_list_not_placed[block_to_be_placed][3:]) + block_to_be_placed
        # decode the action
        idx = self.decode_action(u,self.object_list_not_placed,invalid_idx,block_to_be_placed)
        # retrieve positions of blocks that are to be placed
        curr_pos_to_be_placed = []
        curr_pos_to_be_placed.append(self.object_list_not_placed[block_to_be_placed][:3])
        for i in range(len(self.object_list_not_placed)):
            if ((np.any(invalid_idx==i)) and i!=block_to_be_placed):
                curr_pos_to_be_placed.append(self.object_list_not_placed[i][:3])
        # curr_pos_to_be_placed contains all components that are to be placed
        curr_pos_to_be_placed = np.asarray(curr_pos_to_be_placed).reshape(-1, 3)

        if (self.visualize):
            #remove_block:
            invalid_idx_remove = np.asarray(copy.deepcopy(invalid_idx)).astype(int)
            invalid_idx_remove = np.sort(invalid_idx_remove)
            for i in range(len(invalid_idx_remove)):
                self.yolo_list[invalid_idx_remove[-1-i]].remove()
                self.yolo_list.pop(invalid_idx_remove[-1-i])

        lower_obj = self.object_list[reference_block]
        upper_obj = self.object_list[-1]
        # now place the part
        success = parts_interactions.stack_complex_parts_primitively(lower_obj,upper_obj,idx,self.object_list,self.object_list_not_placed, block_to_be_placed, self.SimEnv.physicsClient)
        # retrieve number of parts that are missing
        missing_samples = self.obj_creator.get_num_remaining_blocks(self.object_list_not_placed)

        if not success:
            print ("ERROR IN THE ENVIRONMENT")
        # to check validity of the action that has been taken -> track velocities of all parts in the scene
        acc_vel_lin1 = np.zeros((len(self.object_list)))
        for j in range(self.num_simulation_steps):
            self.SimEnv.step()
            for i in range(len(self.object_list)):
                curr_lin, curr_ang = self.object_list[i].get_vel()
                acc_vel_lin1[i] += np.sum(np.abs(curr_lin))
        # if hard coded threshold is exceeded -> we have executed an invalid action
        if (np.max(acc_vel_lin1)>0.5):
            prior_exit  = True
        else:
            prior_exit = False
        # if this variable is set we let the part fall until it has settled or a time threshold has been exceeded,...
        if (True):
            if not(prior_exit):
                counter = 0
                threshold = 200
                while(np.abs(self.object_list[-len(invalid_idx)].get_vel()[0][2])>0.1 and counter<threshold):
                    self.SimEnv.step()
                    counter += 1

        if not(prior_exit):
            # check if block is still in shape
            # check if the placed part is still in the same direction as before ('in terms of height as orientation might
            # change due to rotation of the grasp')
            if (len(invalid_idx)!=1):
                curr_pos_to_be_placed_updated = []
                # can only do this check if more than one object
                for i in range((len(invalid_idx))):
                    curr_pos_to_be_placed_updated.append(self.object_list[(-len(invalid_idx)+i)].get_pos_orient()[0])

                curr_pos_to_be_placed_updated = np.asarray(curr_pos_to_be_placed_updated).reshape(-1, 3)
                difference = np.abs(np.subtract(curr_pos_to_be_placed,curr_pos_to_be_placed_updated))[:,2]
                decider = np.abs(difference-np.max(difference))
                if (np.any(decider>0.001)):
                    prior_exit = True
                    # print ("ORIETNATION MUST HAVE CHANGED,...")

            if not (prior_exit):
                # check if other placed blocks are still in right place
                # only then check additionally if parts might have moved (except for the ones that have been placed,...)
                curr_pos_list_updated = []
                for i in range(len(self.object_list) - len(invalid_idx)):
                    curr_pos_list_updated.append(self.object_list[i].get_pos_orient()[0])
                # Calculate distances:
                curr_pos_list_updated = np.asarray(curr_pos_list_updated).reshape(-1, 3)
                distance_arr = np.abs(np.subtract(curr_pos_list, curr_pos_list_updated)) ** 2
                dist_compressed = np.sqrt(distance_arr[:, 0] + distance_arr[:, 1])
                max_movement = np.max(dist_compressed)
                if (max_movement >= 0.01):
                    prior_exit = True

        finished = False
        # if all parts have been placed -> exit
        if (missing_samples==0):
            finished = True

        # special information needed to compute the reward
        placed_parts_obs = self._list_to_array(self._get_list_placed_elements())
        # get reward
        rew, finished1, terminal_rew = self._get_reward(placed_parts_obs)
        # get the observation
        obs = self._get_obs()
        # on any invalid action -> give additional penalty
        if (prior_exit):
            prior_exit = True
            rew = -1.0
        # if successfull -> give additional reward
        elif (finished1):
            rew += 1

        return obs, rew, (finished or finished1 or prior_exit), {}

    def populate_env(self,default_placed=[], default_unplaced=[]):
        default_placed_backup = copy.deepcopy(default_placed)
        default_unplaced_backup = copy.deepcopy(default_unplaced)
        # if env not already empty -> ensure this
        if not(len(self.object_list)==0):
            self.empty_env()
        # if no state to be restored
        if (default_placed==[] and default_unplaced==[]):
            # -> first sample blockset
            list_of_objects = self.obj_creator.sample_blockset(self.num_boxes)
            # place first block randomly
            init_idx = np.random.choice(len(self.target_shapes))
            self.object_list.append(simple_box.Simple_box([self.translation_operations[init_idx][0], self.translation_operations[init_idx][1], 0], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])

            self.list_all_obj = []
            dimensions = self.object_list[0].dimensions
            for i in range(len(list_of_objects)):
                for j in range(len(list_of_objects[i])):
                    # first add scale to the list of objects as they have been initially only defined relatively
                    list_of_objects[i][j][:3] = (np.multiply(np.asarray(list_of_objects[i][j][:3]),2*np.asarray(dimensions))).tolist()
                    self.list_all_obj.append(list_of_objects[i][j])
            # NOW POSITION FIRST OBJECT CORRECTLY:
            parts_interactions.stack_complex_parts_primitively_absolute(self.object_list[0],None,self.list_all_obj,self.object_list,self.SimEnv.physicsClient)
            # retrieve how many other parts have to be placed
            missing_samples = self.obj_creator.get_num_remaining_blocks(self.list_all_obj)
            # place the objects correctly and check if they are in bound
            coord_x = sobol_seq.i4_sobol_generate(1,missing_samples)
            coord_x = (2*coord_x - 1)*(self.observation_high[1]*0.25)

            # place additional blocks behind the one that is placed,...
            self.obj_creator.place_unplaced(self.list_all_obj,self.object_list_not_placed,coord_x,[0,-0.05,0.0])
            unplaced_as_arr = self._list_to_array_idx(self.object_list_not_placed,3).reshape(-1,3)
            # check x,y,z coords <-> if they are still in bounds with respect to the observation space
            check_x_coord = np.sum((np.logical_and(unplaced_as_arr[:,0]>self.observation_low[0], unplaced_as_arr[:,0]<self.observation_high[0]))-1)
            check_y_coord = np.sum((np.logical_and(unplaced_as_arr[:,1]>self.observation_low[1], unplaced_as_arr[:,1]<self.observation_high[1]))-1)
            check_z_coord = np.sum((np.logical_and(unplaced_as_arr[:,2]>self.observation_low[2], unplaced_as_arr[:,2]<self.observation_high[2]))-1)
            if ((check_x_coord+check_y_coord+check_z_coord)!=0):
                print ("CAUTION: ONE OF THE PARTS IS PLACED OUT OF BOUNDS,....")
        else:
            # restore env to state that has been provided to the function
            self.object_list.append(simple_box.Simple_box([default_placed[0][0], default_placed[0][1], default_placed[0][2]-0.05 / 2], [0, 0, 0, 1],self.SimEnv.physicsClient))
            self.object_list[0].change_color([1.0, 0.0, 0.0])
            for i in range(len(default_placed)-1):
                self.object_list.append(
                    simple_box.Simple_box([default_placed[i+1][0], default_placed[i+1][1], default_placed[i+1][2]-0.05 / 2], [0, 0, 0, 1],
                                          self.SimEnv.physicsClient))
            # before placing the unplaced ones -> have to check whether they are close to each other, i.e. whether they
            # together form one object
            unplaced_elements_list = self.obj_creator.decompose_unplaced(copy.deepcopy(default_unplaced), 2*np.asarray(self.object_list[0].dimensions))
            for i in range(len(default_unplaced)):
                self.object_list_not_placed.append([*unplaced_elements_list[i]])

            for i in range(len(self.object_list_not_placed)):
                if (len(self.object_list_not_placed[i][3:])>1):
                    sublist = copy.deepcopy(self.object_list_not_placed[i][3:])
                    sublist.sort()
                    for j in range(len(self.object_list_not_placed[i][3:])):
                        self.object_list_not_placed[i][3+j] = sublist[j]

        # Potential visualization
        self.yolo_list = []
        if (self.visualize):
            # adding offset to not influence the camera!
            for i in range(len(self.object_list_not_placed)):
                self.yolo_list.append(
                    simple_box.Simple_box(
                        [self.object_list_not_placed[i][0], self.object_list_not_placed[i][1]-0.5, 0.0 + self.object_list_not_placed[i][2]],
                        [0, 0, 0, 1],
                        self.SimEnv.physicsClient))
                # if z coordinate is not zero -> we have to add a constraint such that the primitive block does not fall
                # down
                if (self.object_list_not_placed[i][3] != 0.0):
                    import pybullet as p
                    lower_obj = self.yolo_list[-1 + int(self.object_list_not_placed[i][3])]
                    upper_obj = self.yolo_list[-1]
                    cid = p.createConstraint(lower_obj.handle, -1, upper_obj.handle, -1, p.JOINT_FIXED, [0, 0, 0],
                                             upper_obj.get_pos_orient()[0] - lower_obj.get_pos_orient()[0],
                                             [0, 0, 0.0], physicsClientId=lower_obj.physicsClient)

        if (self.check_stability and (default_unplaced_backup==[]) and (default_placed_backup==[])):
            # check stability of structure:
            stability = self.SimEnv.check_initial_stability(self.object_list,self.yolo_list,use_robo=False)
            if (not(stability) and (default_unplaced_backup==[]) and (default_placed_backup==[])):
                # if initial structure is unstable -> we have to populate again,...
                print ("populate again")
                self.populate_env(default_unplaced_backup,default_placed)
