'''
Purpose of this file is to provide the functionality when parts are interacting, e.g. stacking parts on top of each other,
...
'''

import numpy as np
import pybullet as p

def stack_parts_primitively(lower_obj,upper_obj,idx,obj_list):
    # stack parts with respect to each other

    lowerPos, lowerOrn = lower_obj.get_pos_orient()
    # reposition the part that is to be placed baed on the lower object and the idx, where the idx indicates in which
    # direction the part is being placed
    upper_obj.reposition(
        np.asarray(lowerPos) + lower_obj.get_edge_dir()[:, idx] + upper_obj.get_edge_dir_orient(lowerOrn)[:, idx],
        lowerOrn)
    # set the property that there is an object connected now
    lower_obj.child_list[idx][0] = True
    lower_obj.child_list[idx][1] = upper_obj
    if (idx%2==0):
        idx1 = idx+1
    else:
        idx1 = idx -1
    # also change the property of the part that has been placed
    upper_obj.child_list[idx1][0] = True
    upper_obj.child_list[idx1][1] = lower_obj
    return True


def individual_relative_stacking_operation(lower_obj,upper_obj,lower_obj_rel, upper_obj_rel, sim_id):
    # stack two blocks with creating constraint
    lowerPos, lowerOrn = lower_obj.get_pos_orient()
    rel_trafo = upper_obj_rel - lower_obj_rel

    upper_obj.reposition(
        np.asarray(lowerPos) + rel_trafo,
        lowerOrn)
    cid = p.createConstraint(lower_obj.handle, -1, upper_obj.handle, -1, p.JOINT_FIXED, [0, 0, 0],
                             rel_trafo,
                             [0, 0, 0.0], physicsClientId=sim_id)
    return True

from parts import simple_box
def stack_complex_parts_primitively_absolute(lower_obj,upper_obj,obj_list,list_placed,sim_id):
    # to place primitive objects in the right place with respect to each other
    # get initial coordinates of the object
    lowerPos, lowerOrn = lower_obj.get_pos_orient()
    first_box_coords = np.asarray(obj_list[0][:3])
    # dependencies is basically the objects that are connected with each other
    num_dependencies = len(obj_list[0][3:])
    obj_list.pop(0)
    # reposition with respect to the already placed box
    lower_obj.reposition(lowerPos + first_box_coords, lowerOrn)
    # place all the parts correctly
    for i in range(num_dependencies-1):
        list_placed.append(simple_box.Simple_box(
            [0,0,0], [0, 0, 0, 1],
            sim_id))
        curr_box_coords = np.asarray(obj_list[0][:3])
        individual_relative_stacking_operation(lower_obj,list_placed[-1],first_box_coords,curr_box_coords, sim_id)
        obj_list.pop(0)

    return True

def stack_complex_parts_primitively(lower_obj,upper_obj,idx,obj_list,obj_list_unplaced, block_to_be_placed, sim_id):
    # function implements stacking of the complex objects
    removal_list = []
    lowerPos, lowerOrn = lower_obj.get_pos_orient()
    upperPos, upperOrn = upper_obj.get_pos_orient()
    # get coordinates of the block to be placed, and the dependencies,...
    first_box_coords = np.asarray(obj_list_unplaced[block_to_be_placed][:3])
    num_dependencies = len(obj_list_unplaced[block_to_be_placed][3:])
    dependencies = obj_list_unplaced[block_to_be_placed][3:]
    # reposition the object
    upper_obj.reposition(
        np.asarray(lowerPos) + lower_obj.get_edge_dir()[:, idx] + upper_obj.get_edge_dir_orient(lowerOrn)[:, idx],
        lowerOrn)
    removal_list.append(block_to_be_placed)
    # now loop through all the dependencies and also move the other boxes
    for i in range(num_dependencies):
        if (dependencies[i]!=0):
            obj_list.append(simple_box.Simple_box(
                [0,0,0], [0, 0, 0, 1],
                sim_id))
            # get current coords of the block. together with first_box_coords one can compute a relative transformation
            curr_box_coords = np.asarray(obj_list_unplaced[int(block_to_be_placed+dependencies[i])][:3])

            individual_relative_stacking_operation(upper_obj,obj_list[-1],first_box_coords,curr_box_coords, sim_id)
            removal_list.append(block_to_be_placed+dependencies[i])
    removal_list.sort(reverse=True)
    # now remove all the objects before returning
    for i in range(len(removal_list)):
        obj_list_unplaced.pop(int(removal_list[i]))
    return True