import sys
sys.path.append("../")
from env.camera import camera

import pybullet as p
import pybullet_data
import time
import os

import numpy as np

from parts import parts_interactions, simple_box, target_shapes_sampled_points

# this twin environment should mirror the state of the other environment which includes the robot manipulator
# the goal is to avoid having occlusions in the depth image,...
class Standart:

    def __init__(self, visualize=True):
        self.list_elements_placed = []
        self.visualize=visualize
        if (self.visualize):
            self.physicsClient = p.connect(p.GUI_SERVER, 1234, options='--background_color_red=0. --background_color_green=0. --background_color_blue=0.')
            p.resetDebugVisualizerCamera(cameraDistance=1.5999996662139893, cameraPitch=-30.79999923706055, cameraYaw=90,
                                         cameraTargetPosition=[-0.75,0,-0.65+0.025+0.2+0.1], physicsClientId=self.physicsClient)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=self.physicsClient)  # used by loadURDF
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        urdfRootPath = pybullet_data.getDataPath()
        self.planeId = p.loadURDF(os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65],physicsClientId=self.physicsClient)
        self.table_id = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.0, 0.0, -0.65+0.025],physicsClientId=self.physicsClient)

        # define camera list -> easier to use multiple of them
        self.cam_list = []
        self.cam_name_list = []

        self.standart_dt = 1.0/240.0
        # set to 250Hz
        self.change_timestep(1.0/250.0)

        self.log_velocity = False
        self.check_contacts_on_grasp = False
        self.check_table_collision = False
        self.let_part_fall_till_rest = False


    def empty_env(self):
        for i in range(len(self.list_elements_placed)):
            self.list_elements_placed[i].remove()
        self.list_elements_placed.clear()

    def add_camera_static(self,cam_pos, cam_target_pos, cam_up_vector, cam_name, cam_width=512, cam_height=512, fov=120, near=0.25, far=2.0, half_height=False):
        self.cam_list.append(camera.Camera(cam_pos, cam_target_pos, cam_up_vector, cam_width, cam_height, fov, near, far, half_height, simId=self.physicsClient, static=True))
        self.cam_name_list.append(cam_name)

    def add_camera_dynamic(self,rel_cam_pos, cam_target_pos, cam_up_vector, cam_name, related_obj, cam_width=512, cam_height=512, fov=120, near=0.25, far=2.0, half_height=False):
        self.cam_list.append(camera.Camera(rel_cam_pos, cam_target_pos, cam_up_vector, cam_width, cam_height, fov, near, far, half_height, simId=self.physicsClient, static=False, related_obj=related_obj))
        self.cam_name_list.append(cam_name)

    def remove_all_cameras(self):
        self.cam_list = []
        self.cam_name_list = []

    def step(self):
        p.stepSimulation(physicsClientId=self.physicsClient)
        if (self.visualize):
            time.sleep(self.standart_dt)

    def change_timestep(self,time):
        p.setTimeStep(time,physicsClientId=self.physicsClient)
        self.standart_dt = time

    def get_pos_orientation(self, object):
        Pos, Orn = p.getBasePositionAndOrientation(object,physicsClientId=self.physicsClient)
        return Pos, Orn

    def kill(self):
        p.disconnect(physicsClientId=self.physicsClient)

    def populate_env(self,reference_list,num_samples):
        for i in range(num_samples):
            pos, ori = reference_list[-num_samples+i].get_pos_orient()
            self.list_elements_placed.append(simple_box.Simple_box([pos[0], pos[1], pos[2]-0.05 / 2], ori,self.physicsClient))

    # create a rendering of the environment - as we never step this environment we can simply place the parts without
    # actually having to create constraints,...
    def render(self,object_list_placed):
        # check if the list of the objects from the simulated environment and this digital twin are similar
        min_length = min(len(object_list_placed),len(self.list_elements_placed))
        they_are_same = True
        i = 0
        # loop through all elements and check the distances,...
        while(they_are_same and i<min_length):
            distance_arr = np.abs(np.subtract(np.asarray(object_list_placed[i].get_pos_orient()[0]), np.asarray(self.list_elements_placed[i].get_pos_orient()[0]))) ** 2
            if (np.sqrt(np.sum(distance_arr))>0.01):
                they_are_same = False
            else:
                i += 1
        # if they are not the same -> empty the environment and place all elements
        if not(they_are_same):
            self.empty_env()
            self.populate_env(object_list_placed,len(object_list_placed))
        # if they are the same but some elements are missing -> only add the ones that are missing
        elif (they_are_same and (len(object_list_placed)>len(self.list_elements_placed))):
            self.populate_env(object_list_placed, len(object_list_placed)-len(self.list_elements_placed))
        # if they are same but there are less objects -> empty completely and place again,...
        elif (they_are_same and (len(object_list_placed)<len(self.list_elements_placed))):
            # this can also happen on reset
            self.empty_env()
            self.populate_env(object_list_placed,len(object_list_placed))
        # then render all the camera views
        for i in range(len(self.cam_list)):
            self.cam_list[i].render()

    def show_rgb(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_rgb())
        self.cam_list[0].show_multiple_rgb(collect_img,self.cam_name_list)

    def show_rgb_wo_gnd(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_rgb_wo_gnd())
        self.cam_list[0].show_multiple_rgb(collect_img,self.cam_name_list)

    def get_depth(self,cam_id=0, filter=None):
        return self.cam_list[cam_id].get_depth(filter=filter)

    def show_depth(self,filter=None):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_depth(filter=filter))
        self.cam_list[0].show_multiple_depth(collect_img,self.cam_name_list)

    def show_depth_wo_gnd(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_depth_wo_gnd())
        self.cam_list[0].show_multiple_depth(collect_img,self.cam_name_list)

    def show_segment(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_segment())
        self.cam_list[0].show_multiple_rgb(collect_img,self.cam_name_list)