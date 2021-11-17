# The purpose of this file is to sample complex objects and initialize them correctly,...


import sys
sys.path.append("../")
import random
import numpy as np
import copy
import random
from scipy.spatial.transform import Rotation as R

def process_shapes(input):
    # the return is always a list with properties as follows: realtive x,y,z coordinates and an array with reference to the other parts,..
    if (input[0]=='horizontal'):
        return horizontal_group(input[1])
    elif (input[0]=='vertical'):
        return vertical_group(input[1])
    elif (input[0]=='horizontal_random'):
        return horizontal_group_random(input[1])
    elif (input[0]=='square'):
        return square_group(input[1])

    elif (input[0]=='l_ground'):
        return l_ground_random(input[1])
    elif (input[0]=='l_up'):
        return l_up_random(input[1])
    elif (input[0]=='w_ground'):
        return w_ground_random(input[1])
    elif (input[0]=='w_up'):
        return w_up_random(input[1])
    elif (input[0]=='block_2_2_random'):
        return block_2_2_random(input[1])
    elif (input[0]=='block_s_random'):
        return block_s_random(input[1])

    else:
        print ("UNKNOWN SHAPE DEMANDED -> ERROR!!!")



def horizontal_group_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return horizontal_group(num_elements,rotation=rotation)

def horizontal_group(num_elements,rotation=[0,0,0]):
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    r = R.from_rotvec(np.asarray(rotation))
    for i in range(num_elements):
        coord = np.matmul(r.as_matrix(),np.asarray([i,0,0]))
        return_list.append([*(coord).tolist(),*(reference_to_others-i).tolist()])
    # randomize order of blocks to remove bias of structure (alternative would be to control direction,...)
    return_list_copied = copy.deepcopy(return_list)
    random.shuffle(return_list)
    # but keep the references,...
    for i in range(len(return_list)):
        return_list[i][3:] = return_list_copied[i][3:]
    return return_list

def vertical_group(num_elements):
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    for i in range(num_elements):
        return_list.append([0,0,i,*(reference_to_others-i).tolist()])
    return return_list

def square_group(num_elements):
    if (num_elements!=1):
        print ("ERROR IN SQUARE GROUP -> MORE THAN 1 ELEMENT ATM NOT SUPPORTED,..")
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    for i in range(num_elements):
        return_list.append([0,0,i,*(reference_to_others-i).tolist()])
    return return_list


def l_ground_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return l_ground(num_elements,rotation=rotation)

def l_ground(num_elements,rotation=[0,0,0]):
    if (num_elements<3):
        print ("L shape does not work -> abort!!!")
    r = R.from_rotvec(np.asarray(rotation))
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 0).tolist()])
    for i in range(num_elements-1):
        coord = np.matmul(r.as_matrix(), np.asarray([0, i, 0]))
        return_list.append([*(coord).tolist(),*(reference_to_others-(i+1)).tolist()])
    return return_list

def l_up_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return l_up(num_elements,rotation=rotation)

def l_up(num_elements,rotation=[0,0,0]):
    if (num_elements<3):
        print ("L shape does not work -> abort!!!")
    r = R.from_rotvec(np.asarray(rotation))
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 0).tolist()])
    for i in range(num_elements-1):
        coord = np.matmul(r.as_matrix(), np.asarray([0, 0, i]))
        return_list.append([*(coord).tolist(),*(reference_to_others-(i+1)).tolist()])
    return return_list

def w_ground_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return w_ground(num_elements,rotation=rotation)

def w_ground(num_elements,rotation=[0,0,0]):
    if (num_elements<4):
        print ("W shape does not work -> abort!!!")
    r = R.from_rotvec(np.asarray(rotation))
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)

    #pplace first 3 elements hardcoded:
    coord = np.matmul(r.as_matrix(), np.asarray([-1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - (0)).tolist()])
    coord = np.matmul(r.as_matrix(), np.asarray([0, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - (1)).tolist()])
    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - (2)).tolist()])

    for i in range(num_elements-3):
        coord = np.matmul(r.as_matrix(), np.asarray([0, (i+1), 0]))
        return_list.append([*(coord).tolist(),*(reference_to_others-(i+3)).tolist()])
    return return_list

def w_up_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return w_up(num_elements,rotation=rotation)

def w_up(num_elements,rotation=[0,0,0]):
    if (num_elements<4):
        print ("W shape does not work -> abort!!!")
    r = R.from_rotvec(np.asarray(rotation))
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)

    #pplace first 3 elements hardcoded:
    coord = np.matmul(r.as_matrix(), np.asarray([-1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - (0)).tolist()])
    coord = np.matmul(r.as_matrix(), np.asarray([0, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - (1)).tolist()])
    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - (2)).tolist()])

    for i in range(num_elements-3):
        coord = np.matmul(r.as_matrix(), np.asarray([0, 0, (i+1)]))
        return_list.append([*(coord).tolist(),*(reference_to_others-(i+3)).tolist()])
    return return_list

def block_2_2_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return block_2_2(num_elements,rotation=rotation)

def block_2_2(num_elements,rotation=[0,0,0]):
    if (num_elements!=4):
        print ("block does not work -> abort!!!")
    r = R.from_rotvec(np.asarray(rotation))
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 0).tolist()])

    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 1]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 1).tolist()])

    coord = np.matmul(r.as_matrix(), np.asarray([0, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 2).tolist()])

    coord = np.matmul(r.as_matrix(), np.asarray([0, 0, 1]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 3).tolist()])
    return return_list


def block_s_random(num_elements):
    rand_numb = random.randint(0,3)
    rotation = [0,0,np.pi / 2*rand_numb]
    return block_s(num_elements,rotation=rotation)

def block_s(num_elements,rotation=[0,0,0]):
    if (num_elements!=4):
        print ("block does not work -> abort!!!")
    r = R.from_rotvec(np.asarray(rotation))
    return_list = []
    reference_to_others = np.linspace(0,num_elements,num=num_elements, endpoint=False)
    coord = np.matmul(r.as_matrix(), np.asarray([1, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 0).tolist()])

    coord = np.matmul(r.as_matrix(), np.asarray([0, 0, 0]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 1).tolist()])

    coord = np.matmul(r.as_matrix(), np.asarray([0, 0, 1]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 2).tolist()])

    coord = np.matmul(r.as_matrix(), np.asarray([-1, 0, 1]))
    return_list.append([*(coord).tolist(), *(reference_to_others - 3).tolist()])
    return return_list



class ComplexObjects:

    def __init__(self):
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.shapes_available.append(['horizontal',2])
        self.shapes_available.append(['vertical',2])
        self.shapes_available.append(['square',1])

        self.selected_shapes = []

    def set_num_blocks(self, num_blocks):
        self.num_blocks_available = num_blocks

    def sample_blockset(self,num_blocks):
        self.set_num_blocks(num_blocks)
        self.selected_shapes = []

        parts_used = 0
        half_num_blocks = num_blocks // 2
        num_samples = np.random.randint((half_num_blocks//2)+1)
        # Step 1: select number of primitve shapes to be used
        for i in range(num_samples):
            self.selected_shapes.append(self.shapes_available[0])
            parts_used += self.selected_shapes[-1][1]
        num_samples = np.random.randint((half_num_blocks//2)+1)
        for i in range(num_samples):
            self.selected_shapes.append(self.shapes_available[1])
            parts_used += self.selected_shapes[-1][1]

        parts_remaining = self.num_blocks_available - parts_used
        for i in range(parts_remaining):
            self.selected_shapes.append(self.shapes_available[2])
            parts_used += self.selected_shapes[-1][1]

        if (parts_used!=self.num_blocks_available):
            print ("ERROR IN CHOOSING THE SHAPES,..")

        # RANDOMIZE THE ORDERING OF THE SHAPES,...
        random.shuffle(self.selected_shapes)

        # Step 2: create list that can be processed by outside function,...
        return_list = []
        for i in range(len(self.selected_shapes)):
            return_list.append(process_shapes(self.selected_shapes[i]))

        return return_list

    def get_num_remaining_blocks(self,input_list):
        if (len(input_list)==0):
            return 0
        num_elements = 0
        current_idx = 0
        len_list = len(input_list)
        while True:
            num_elements += 1
            current_idx += len(input_list[current_idx][3:])
            if (current_idx>=len_list):
                break
        return num_elements

    def convert_to_array(self,list):
        empty_one = []
        for i in range(len(list)):
            empty_one.append([*list[i][:3]])

        return np.asarray(empty_one).reshape(-1,3)

    def check_collision_and_adapt(self,to_be_placed, currently_placed):
        additional_increment = np.asarray([0,-0.05,0])

        while(True):
            in_collision = False

            to_be_placed_arr = self.convert_to_array(to_be_placed)

            if not(len(currently_placed)==0):
                currently_placed_arr = self.convert_to_array(currently_placed)
                for i in range(len(to_be_placed)):
                    if not(in_collision):
                        dist_arr = np.subtract(currently_placed_arr,to_be_placed_arr[i,:])
                        dist_arr = np.sqrt(np.sum(dist_arr**2,axis=1))

                        if (np.any(dist_arr<0.07)):
                            in_collision = True

            # check y coordinate
            if (np.any(to_be_placed_arr[:,1]>-0.075)):
                in_collision = True

            if (in_collision):
                for i in range(len(to_be_placed)):
                    to_be_placed[i][0] += additional_increment[0]
                    to_be_placed[i][1] += additional_increment[1]
                    to_be_placed[i][2] += additional_increment[2]

            else:
                break

    # place an unplaced block in the scene
    def place_unplaced(self, input_list, unplaced_list, sobol_pos, global_offset):
        first_elem_idx = 0
        num_first_elements = 0
        total_num_blocks = len(input_list)

        list_to_be_added = []
        for i in range(len(input_list)):
            if (first_elem_idx==i):
                if (i!=0):
                    # before adding -> do collision checking!
                    self.check_collision_and_adapt(list_to_be_added,unplaced_list)

                    unplaced_list.extend(list_to_be_added)
                    list_to_be_added = []

                first_pos_global = (np.asarray(input_list[i][:3]) + np.asarray([sobol_pos[num_first_elements,0]+global_offset[0],global_offset[1],global_offset[2]]))
                first_pos_local = np.asarray(input_list[i][:3])
                list_to_be_added.append([*first_pos_global.tolist(),*input_list[i][3:]])
                first_elem_idx += len(input_list[i][3:])
                num_first_elements += 1
            else:
                curr_pos_local = np.asarray(input_list[i][:3])
                offset = curr_pos_local-first_pos_local
                list_to_be_added.append([*(first_pos_global+offset).tolist(),*input_list[i][3:]])

            if (i==total_num_blocks-1):
                self.check_collision_and_adapt(list_to_be_added, unplaced_list)
                unplaced_list.extend(list_to_be_added)
                break


    # decompose the list of unplaced elements into a list that includes the connectivity information
    def decompose_unplaced(self, input_list, dimensions):
        return_list = []
        newly_added = []
        curr_to_be_added = []

        while(len(input_list)!=0 or len(newly_added)!=0):
            if (len(newly_added)==0):
                # proceed with normal list
                curr_sample = input_list.pop(0)
                curr_to_be_added.append([*curr_sample, 0])

            else:
                curr_sample = newly_added.pop(0)

            curr_input_list_as_arr = np.asarray(input_list)
            if (len(input_list)!=0):
                # check all vor vicinity:
                vicinity_x = np.abs(np.abs(curr_input_list_as_arr[:,0]-curr_sample[0]) - dimensions[0]) < 0.001
                vicinity_y = np.abs(np.abs(curr_input_list_as_arr[:,1]-curr_sample[1]) - dimensions[1]) < 0.001
                vicinity_z = np.abs(np.abs(curr_input_list_as_arr[:,2]-curr_sample[2]) - dimensions[2]) < 0.001
                # check for equality:
                equal_x = np.abs(curr_input_list_as_arr[:,0]-curr_sample[0]) < 0.001
                equal_y = np.abs(curr_input_list_as_arr[:,1]-curr_sample[1]) < 0.001
                equal_z = np.abs(curr_input_list_as_arr[:,2]-curr_sample[2]) < 0.001

                general_vicinity = vicinity_x.astype(int)+vicinity_y.astype(int)+vicinity_z.astype(int)
                general_vicinity[general_vicinity==3] = -1 # this is unwanted
                general_equality = equal_x.astype(int) + equal_y.astype(int) + equal_z.astype(int)
                general_equality[general_equality == 3] = -1  # this is unwanted

                combined = general_equality + general_vicinity
                decider = np.where(combined==3)
            else:
                decider = [[]]

            if (np.shape(decider)[1]==0):
                if (len(newly_added)==0):
                    # only then add them, otherwise proceed,...
                    return_list.extend(copy.deepcopy(curr_to_be_added))
                    curr_to_be_added = []
            else:
                idx_list = []
                for j in range(len(decider)):
                    curr_idx = decider[j]
                    curr_idx = curr_idx[0]
                    idx_list.append(curr_idx)
                    curr_to_be_added.append([*input_list[curr_idx], 0])
                    newly_added.append([*input_list[curr_idx]])
                    for i in range(len(curr_to_be_added)):
                        if (i==len(curr_to_be_added)-1):
                            to_append = (-1*np.linspace(1,i,num=i)).tolist()
                            curr_to_be_added[i] = [*curr_to_be_added[i],*to_append]
                        else:
                            curr_to_be_added[i] = [*curr_to_be_added[i], len(curr_to_be_added) - 1 - i]


                idx_list.sort(reverse=True)
                for i in range(len(idx_list)):
                    input_list.pop(idx_list[i])

        return return_list

# mainly contains a different set of blocks
class ComplexObjects1(ComplexObjects):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',3])
        self.sub_list.append(['horizontal_random',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['vertical',3])
        self.sub_list.append(['vertical',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.selected_shapes = []

    # this is also a new, more randomized block selection strategy
    def sample_blockset(self,num_blocks):
        self.set_num_blocks(num_blocks)
        self.selected_shapes = []

        parts_used = 0

        while(True):
            if (self.num_blocks_available-parts_used)==1:
                self.selected_shapes.append(self.shapes_available[0][0])
                parts_used += self.selected_shapes[-1][1]
                break

            # else do a random selector
            rand_numb = random.randint(0, len(self.shapes_available)-1)
            if (len(self.shapes_available[rand_numb])!=1):
                # we have to do additional selection
                rand2 = random.randint(0, int(len(self.shapes_available[rand_numb]))-1)
                potential_new = self.shapes_available[rand_numb][rand2]
            else:
                potential_new = self.shapes_available[rand_numb][0]


            if ((parts_used+potential_new[1]==self.num_blocks_available)):
                self.selected_shapes.append(potential_new)
                parts_used += self.selected_shapes[-1][1]
                break
            elif ((parts_used+potential_new[1]<=self.num_blocks_available)):
                self.selected_shapes.append(potential_new)
                parts_used += self.selected_shapes[-1][1]

        if (parts_used!=self.num_blocks_available):
            print ("ERROR IN CHOOSING THE SHAPES,..")

        # RANDOMIZE THE ORDERING OF THE SHAPES,...
        random.shuffle(self.selected_shapes)

        # Step 2: create list that can be processed by outside function,...
        return_list = []
        for i in range(len(self.selected_shapes)):
            return_list.append(process_shapes(self.selected_shapes[i]))

        return return_list


# another set of objects
class ComplexObjects2(ComplexObjects1):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',3])
        self.sub_list.append(['horizontal_random',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['vertical',3])
        self.sub_list.append(['vertical',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.sub_list.append(['l_ground',3])
        self.sub_list.append(['w_up',4])
        self.sub_list.append(['w_ground',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.selected_shapes = []

# another set of objects
class ComplexObjects10(ComplexObjects2):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',3])
        self.sub_list.append(['horizontal_random',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.sub_list.append(['l_ground',3])
        self.sub_list.append(['w_up',4])
        self.sub_list.append(['w_ground',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.selected_shapes = []

# another set of blocks which tries to mimic the placing for the robot (on one side of the table)
class ComplexObjects3(ComplexObjects1):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.selected_shapes = []

    def place_unplaced(self, input_list, unplaced_list, sobol_pos, global_offset):
        first_elem_idx = 0
        num_first_elements = 0
        total_num_blocks = len(input_list)

        list_to_be_added = []
        for i in range(len(input_list)):
            if ((i+1)%6==0):
                global_offset[0] -= 0.025
                global_offset[1] -= 0.1
            if (first_elem_idx==i):
                if (i!=0):
                    # before adding -> do collision checking!
                    self.check_collision_and_adapt(list_to_be_added,unplaced_list)


                    unplaced_list.extend(list_to_be_added)
                    list_to_be_added = []

                first_pos_global = (np.asarray(input_list[i][:3]) + np.asarray([sobol_pos[num_first_elements,0]+global_offset[0],global_offset[1],global_offset[2]]))
                first_pos_local = np.asarray(input_list[i][:3])
                # first_pos_global[1] = -first_pos_global[1]
                list_to_be_added.append([*first_pos_global.tolist(),*input_list[i][3:]])
                first_elem_idx += len(input_list[i][3:])
                num_first_elements += 1
            else:
                curr_pos_local = np.asarray(input_list[i][:3])
                offset = curr_pos_local-first_pos_local
                list_to_be_added.append([*(first_pos_global+offset).tolist(),*input_list[i][3:]])

            if (i==total_num_blocks-1):
                self.check_collision_and_adapt(list_to_be_added, unplaced_list)
                unplaced_list.extend(list_to_be_added)
                break

    def check_collision_and_adapt(self,to_be_placed, currently_placed):
        additional_increment = np.asarray([0,-0.05,0])

        while(True):
            in_collision = False

            to_be_placed_arr = self.convert_to_array(to_be_placed) #np.asarray(to_be_placed)[:,:3]

            if not(len(currently_placed)==0):
                # currently_placed_arr = np.asarray(currently_placed)[:, :3]
                currently_placed_arr = self.convert_to_array(currently_placed)
                for i in range(len(to_be_placed)):
                    if not(in_collision):
                        dist_arr = np.subtract(currently_placed_arr,to_be_placed_arr[i,:])
                        dist_arr = np.sqrt(np.sum(dist_arr**2,axis=1))

                        if (np.any(dist_arr<0.07)):
                            in_collision = True

            # check y coordinate
            if (np.any(to_be_placed_arr[:,1]>-0.075)):
                in_collision = True

            if (in_collision):
                for i in range(len(to_be_placed)):
                    to_be_placed[i][0] += additional_increment[0]
                    to_be_placed[i][1] += additional_increment[1]
                    to_be_placed[i][2] += additional_increment[2]

            else:
                break

# place the objects completely at random
class ComplexObjects4(ComplexObjects1):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        # print ("HALLO")
        self.shapes_available = [] # defines which shapes are available
        # self.shape_complexity = [] # defines complexity of shape, i.e. how many blocks it contains,..

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.selected_shapes = []



    def place_unplaced(self, input_list, unplaced_list, sobol_pos, global_offset):
        first_elem_idx = 0
        num_first_elements = 0
        total_num_blocks = len(input_list)

        self.x_limit = 0.3843749910593033 * 0.75
        self.x_limit_bias = -0.07
        self.y_limit_low = 0.15
        self.y_limit_high = 0.35

        list_to_be_added = []
        for i in range(len(input_list)):
            if (first_elem_idx==i):
                # determine new global offset at random:
                global_offset[0] = (2*np.random.rand()-1)*self.x_limit + self.x_limit_bias
                global_offset[1] = np.sign((2*np.random.rand()-1)) * (self.y_limit_low + np.random.rand()*(self.y_limit_high-self.y_limit_low))

                if (i!=0):
                    # before adding -> do collision checking!
                    self.check_collision_and_adapt(list_to_be_added,unplaced_list)


                    unplaced_list.extend(list_to_be_added)
                    list_to_be_added = []

                first_pos_global = (np.asarray(input_list[i][:3]) + np.asarray([global_offset[0],global_offset[1],global_offset[2]]))
                first_pos_local = np.asarray(input_list[i][:3])
                # first_pos_global[1] = -first_pos_global[1]
                list_to_be_added.append([*first_pos_global.tolist(),*input_list[i][3:]])
                first_elem_idx += len(input_list[i][3:])
                num_first_elements += 1
            else:
                curr_pos_local = np.asarray(input_list[i][:3])
                offset = curr_pos_local-first_pos_local
                list_to_be_added.append([*(first_pos_global+offset).tolist(),*input_list[i][3:]])

            if (i==total_num_blocks-1):
                self.check_collision_and_adapt(list_to_be_added, unplaced_list)
                unplaced_list.extend(list_to_be_added)
                break

    def check_collision_and_adapt(self,to_be_placed, currently_placed):
        additional_increment = np.asarray([0,-0.05,0])

        while(True):
            in_collision = False

            to_be_placed_arr = self.convert_to_array(to_be_placed) #np.asarray(to_be_placed)[:,:3]

            if not(len(currently_placed)==0):
                # currently_placed_arr = np.asarray(currently_placed)[:, :3]
                currently_placed_arr = self.convert_to_array(currently_placed)
                for i in range(len(to_be_placed)):
                    if not(in_collision):
                        dist_arr = np.subtract(currently_placed_arr,to_be_placed_arr[i,:])
                        dist_arr = np.sqrt(np.sum(dist_arr**2,axis=1))

                        if (np.any(dist_arr<0.095)): # originally 0.07
                            in_collision = True

            # check y coordinate
            if (np.any(to_be_placed_arr[:,1]<0)):
                if (np.any(to_be_placed_arr[:,1]>-0.075) or np.all(np.abs(to_be_placed_arr[:,1])>self.y_limit_high)):
                    in_collision = True
                if (np.all(np.abs(to_be_placed_arr[:,1])>self.y_limit_high)):
                    reposition = True
                else:
                    reposition = False

                if (reposition):
                    new_x = (2 * np.random.rand() - 1) * self.x_limit + self.x_limit_bias
                    new_y = np.sign((2 * np.random.rand() - 1)) * (
                            self.y_limit_low + np.random.rand() * (self.y_limit_high - self.y_limit_low))


                    # place first element at the position
                    to_be_placed[0][0] = new_x
                    to_be_placed[0][1] = new_y

                    for i in range(len(to_be_placed)-1):
                        # new pos = new one + relative trafo
                        to_be_placed[i+1][0] = to_be_placed[i][0] + (to_be_placed_arr[i+1,0]-to_be_placed_arr[i,0])
                        to_be_placed[i+1][1] = to_be_placed[i][1] + (to_be_placed_arr[i+1,1]-to_be_placed_arr[i,1])

                        # to_be_placed[i][0]= (2 * np.random.rand() - 1) * self.x_limit + self.x_limit_bias
                        # to_be_placed[i][1] = np.sign((2 * np.random.rand() - 1)) * (
                        #             self.y_limit_low + np.random.rand() * (self.y_limit_high - self.y_limit_low))

                elif (in_collision):
                    for i in range(len(to_be_placed)):
                        to_be_placed[i][0] += additional_increment[0]
                        to_be_placed[i][1] += additional_increment[1]
                        to_be_placed[i][2] += additional_increment[2]


                else:
                    break

            # check y coordinate
            else:
                if (np.any(to_be_placed_arr[:, 1] < 0.075)):
                    in_collision = True

                if (np.all(np.abs(to_be_placed_arr[:,1])>self.y_limit_high)):
                    reposition = True
                else:
                    reposition = False

                if (reposition):
                    new_x = (2 * np.random.rand() - 1) * self.x_limit + self.x_limit_bias
                    new_y = np.sign((2 * np.random.rand() - 1)) * (
                            self.y_limit_low + np.random.rand() * (self.y_limit_high - self.y_limit_low))


                    # place first element at the position
                    to_be_placed[0][0] = new_x
                    to_be_placed[0][1] = new_y

                    for i in range(len(to_be_placed)-1):
                        # new pos = new one + relative trafo
                        to_be_placed[i+1][0] = to_be_placed[i][0] + (to_be_placed_arr[i+1,0]-to_be_placed_arr[i,0])
                        to_be_placed[i+1][1] = to_be_placed[i][1] + (to_be_placed_arr[i+1,1]-to_be_placed_arr[i,1])


                elif (in_collision):
                    for i in range(len(to_be_placed)):
                        to_be_placed[i][0] -= additional_increment[0]
                        to_be_placed[i][1] -= additional_increment[1]
                        to_be_placed[i][2] -= additional_increment[2]

                else:
                    break


# more complex set of blocks, but place them again at only one side
# NOTE: in the current implementation when sampling a random blockset, the first element is always a simple square. In case
# this should be adapted, one also has to modify the procedure "stack_complex_parts_primitively_absolute" inside parts_interactions.py
class ComplexObjects20(ComplexObjects1):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',3])
        self.sub_list.append(['horizontal_random',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['vertical',3])
        self.sub_list.append(['vertical',2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.sub_list.append(['l_ground',3])
        self.sub_list.append(['w_up',4])
        self.sub_list.append(['w_ground',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))


    def sample_blockset(self,num_blocks):
        self.set_num_blocks(num_blocks)
        self.selected_shapes = []

        # always start with 1 red block,... (see also line 838)
        parts_used = 1

        while(True):
            if (self.num_blocks_available-parts_used)==1:
                self.selected_shapes.append(self.shapes_available[0][0])
                parts_used += self.selected_shapes[-1][1]
                break

            # else do a random selector
            rand_numb = random.randint(0, len(self.shapes_available)-1)
            if (len(self.shapes_available[rand_numb])!=1):
                # we have to do additional selection
                rand2 = random.randint(0, int(len(self.shapes_available[rand_numb]))-1)
                potential_new = self.shapes_available[rand_numb][rand2]
            else:
                potential_new = self.shapes_available[rand_numb][0]


            if ((parts_used+potential_new[1]==self.num_blocks_available)):
                self.selected_shapes.append(potential_new)
                parts_used += self.selected_shapes[-1][1]
                break
            elif ((parts_used+potential_new[1]<=self.num_blocks_available)):
                self.selected_shapes.append(potential_new)
                parts_used += self.selected_shapes[-1][1]


        if (parts_used!=self.num_blocks_available):
            print ("ERROR IN CHOOSING THE SHAPES,..")

        # RANDOMIZE THE ORDERING OF THE SHAPES,...
        random.shuffle(self.selected_shapes)
        self.selected_shapes.insert(0,['square',1])

        # Step 2: create list that can be processed by outside function,...
        return_list = []
        for i in range(len(self.selected_shapes)):
            return_list.append(process_shapes(self.selected_shapes[i]))

        return return_list

# again simply a different set of blocks selected
class ComplexObjects21(ComplexObjects20):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',2])
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.sub_list.append(['l_ground',3])

        self.shapes_available.append(copy.deepcopy(self.sub_list))

# again simply a different set of blocks selected
class ComplexObjects22(ComplexObjects20):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',2])
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.sub_list.append(['l_ground',3])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['block_2_2_random',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['block_s_random',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

# again simply a different set of blocks selected
class ComplexObjects23(ComplexObjects20):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['horizontal_random',2])
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.sub_list.append(['l_ground',3])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['block_s_random',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

# again simply a different set of blocks selected
class ComplexObjects25(ComplexObjects20):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['square', 1])
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])

        self.shapes_available.append(copy.deepcopy(self.sub_list))

# again simply a different set of blocks selected
class ComplexObjects26(ComplexObjects20):

    def __init__(self):
        super().__init__()
        self.num_blocks_available = None
        self.shapes_available = [] # defines which shapes are available

        self.sub_list = []
        self.sub_list.append(['square',1])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['square', 1])
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['square', 1])
        self.sub_list.append(['vertical', 2])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['l_up',3])
        self.shapes_available.append(copy.deepcopy(self.sub_list))

        self.sub_list = []
        self.sub_list.append(['block_s_random',4])
        self.shapes_available.append(copy.deepcopy(self.sub_list))