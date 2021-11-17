import sys
sys.path.append("../")
from env.camera import camera

import pybullet as p
import pybullet_data
import time
import os

import numpy as np
from numpy.linalg import norm, solve

import pinocchio
import copy

# This file implements the standard environment for the assembly

class Standart:

    def __init__(self, visualize=True):
        self.visualize=visualize
        if (self.visualize):
            # if yes - enable visualization and place camera properly
            self.physicsClient = p.connect(p.GUI_SERVER, 1234, options='--background_color_red=0. --background_color_green=0. --background_color_blue=0.')
            p.resetDebugVisualizerCamera(cameraDistance=1.5999996662139893, cameraPitch=-30.79999923706055, cameraYaw=75,
                                         cameraTargetPosition=[-0.75,0,-0.65+0.025+0.2+0.1], physicsClientId=self.physicsClient)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(),physicsClientId=self.physicsClient)  # used by loadURDF
        p.setGravity(0, 0, -10, physicsClientId=self.physicsClient)
        self.planeId = p.loadURDF("plane.urdf", physicsClientId=self.physicsClient)

        # define available cameras as list -> easier to use multiple of them
        self.cam_list = []
        self.cam_name_list = []

        # standart timestep of pybullet is set to 240 Hz
        self.standart_dt = 1.0/240.0
        # change timestep to 250Hz
        self.change_timestep(1.0/250.0)

        # define some basic booleans
        self.log_velocity = False
        self.check_contacts_on_grasp = False
        self.check_table_collision = False
        self.let_part_fall_till_rest = False


    # 3 camera related comfort functions, i.e. adding and removing cameras
    def add_camera_static(self,cam_pos, cam_target_pos, cam_up_vector, cam_name, cam_width=512, cam_height=512, fov=120, near=0.25, far=2.0, half_height=False):
        self.cam_list.append(camera.Camera(cam_pos, cam_target_pos, cam_up_vector, cam_width, cam_height, fov, near, far, half_height, simId=self.physicsClient, static=True))
        self.cam_name_list.append(cam_name)

    def add_camera_dynamic(self,rel_cam_pos, cam_target_pos, cam_up_vector, cam_name, related_obj, cam_width=512, cam_height=512, fov=120, near=0.25, far=2.0, half_height=False):
        self.cam_list.append(camera.Camera(rel_cam_pos, cam_target_pos, cam_up_vector, cam_width, cam_height, fov, near, far, half_height, simId=self.physicsClient, static=False, related_obj=related_obj))
        self.cam_name_list.append(cam_name)

    def remove_all_cameras(self):
        self.cam_list = []
        self.cam_name_list = []


    # this function steps the simulation - in case of visualization we also do an appropriate sleep,...
    def step(self):
        p.stepSimulation(physicsClientId=self.physicsClient)
        if (self.visualize):
            time.sleep(self.standart_dt)

    # functionality to change the timestep
    def change_timestep(self,time):
        p.setTimeStep(time,physicsClientId=self.physicsClient)
        self.standart_dt = time

    # getting the position and orientation of an arbitrary object in the scene
    def get_pos_orientation(self, object):
        Pos, Orn = p.getBasePositionAndOrientation(object,physicsClientId=self.physicsClient)
        return Pos, Orn

    # killing the simulation environment
    def kill(self):
        p.disconnect(physicsClientId=self.physicsClient)

    # this function renders the camera views of the scene
    def render(self):
        for i in range(len(self.cam_list)):
            self.cam_list[i].render()

    # function to illustrate a rgb image of the scene
    def show_rgb(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_rgb())
        self.cam_list[0].show_multiple_rgb(collect_img,self.cam_name_list)

    # function to illustrate a rgb image of the scene, however with eliminating the ground plane
    def show_rgb_wo_gnd(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_rgb_wo_gnd())
        self.cam_list[0].show_multiple_rgb(collect_img,self.cam_name_list)

    # function gets the depth image of a specific camera
    def get_depth(self,cam_id=0, filter=None):
        return self.cam_list[cam_id].get_depth(filter=filter)

    # show the depth images of all cameras
    def show_depth(self,filter=None):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_depth(filter=filter))
        self.cam_list[0].show_multiple_depth(collect_img,self.cam_name_list)

    # show depth images of all cameras, however under removing the ground plane,..
    def show_depth_wo_gnd(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_depth_wo_gnd())
        self.cam_list[0].show_multiple_depth(collect_img,self.cam_name_list)

    # show the segmentation masks of the scene
    def show_segment(self):
        collect_img = []
        for i in range(len(self.cam_list)):
            collect_img.append(self.cam_list[i].get_segment())
        self.cam_list[0].show_multiple_rgb(collect_img,self.cam_name_list)

    # this function adds the lab setup, i.e. the robot and co,...
    def add_lab_setup(self):
        self.optimize_time = True
        if (self.optimize_time):
            self.use_constraint = True # use a constraint to grasp the parts - this allows to move parts faster
        else:
            self.use_constraint = True # this might be also set to false to make things more realistic, however,
                                       # especially with the more complex objects, it necessarily has to be set to true
                                       # as the bigger parts are also only held together through constraints and might
                                       # otherwise fall apart

        self.ee_signal = [0.08,0.08]
        p.removeBody(self.planeId, physicsClientId=self.physicsClient)
        urdfRootPath = pybullet_data.getDataPath()
        # load robot into the scene
        self.robot_id = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True, basePosition=[-0.5, 0, 0.0],physicsClientId=self.physicsClient)
        # reset it
        self.reset_robot()
        # load the table
        self.table_id = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.0, 0.0, -0.65+0.025],physicsClientId=self.physicsClient)
        # load some robot properties
        self.n_joints = p.getNumJoints(self.robot_id,physicsClientId=self.physicsClient)
        self.joint_infos = [p.getJointInfo(self.robot_id, i, physicsClientId=self.physicsClient) for i in range(self.n_joints)]
        self.joint_index = {info[1].decode(): info[0] for info in self.joint_infos}

        self.tool_link = 11#link_from_name(self.robot_id, 'panda_grasptarget')

        # in case of optimizing the time, we set some specific properties for the collisions
        if (self.optimize_time):
            group = 0  # other objects don't collide with me
            mask = 0  # don't collide with any other object
            for i in range(8):
                p.setCollisionFilterGroupMask(self.robot_id, i, group, mask,physicsClientId=self.physicsClient)
        # in case of wanting to optimize time, the number of optimization steps change
        if (self.optimize_time):
            self.num_steps_planning = int(50*1.0)
        else:
            self.num_steps_planning = 50 * 4

        # determine if the output should be printed or not
        self.verbose=False
        # load required components for pinocchio
        urdfRootPath = pybullet_data.getDataPath()
        self.model = pinocchio.buildModelFromUrdf(os.path.join(urdfRootPath, "franka_panda/panda.urdf"))
        self.data = None
        self.frameId = self.model.getFrameId("panda_grasptarget")


    # function to reset the robot to some desired configuration
    def reset_robot(self,poses=None,velocities=None):
        if (poses is None):
            poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        if (velocities is None):
            velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for i in range(7):
            p.resetJointState(self.robot_id, i, poses[i], velocities[i], physicsClientId=self.physicsClient)
        p.resetJointState(self.robot_id, 9, poses[7], velocities[7], physicsClientId=self.physicsClient)
        p.resetJointState(self.robot_id, 10, poses[8], velocities[8], physicsClientId=self.physicsClient)
        self.q = np.asarray(poses)
        self.step()
        if (self.log_velocity):
            self.do_log_velocity()

    # returns information about the robot, i.e. joint poses and velocities
    def get_robot_info(self):
        poses = []
        velocities = []
        for i in range(7):
            info = p.getJointState(self.robot_id, i, physicsClientId=self.physicsClient)
            poses.append(info[0])
            velocities.append(info[1])
        info = p.getJointState(self.robot_id, 9, physicsClientId=self.physicsClient)
        poses.append(info[0])
        velocities.append(info[1])
        info = p.getJointState(self.robot_id, 10, physicsClientId=self.physicsClient)
        poses.append(info[0])
        velocities.append(info[1])

        return poses, velocities

    # log the velocities of all components included in the logging list
    def do_log_velocity(self):
        for i in range(len(self.logging_list)):
            curr_lin, curr_ang = self.logging_list[i].get_vel()
            self.logging_arr[i] += np.sum(np.abs(curr_lin))

    # function to grasp a part that includes checking the velocity of all other parts in the scene to ensure that during
    # the grasping process we do not bump into any other object
    def grasp_part_w_vel_check(self,pos,ori,obj_to_be_moved,list_unplaced,idx_to_be_moved,initial_pose=None,slow_down=False):
        # boolean slow_down is useful if you want to have fast execution in all threads / simulations in which MCTS is
        # conducted but slower speed in the environment which is actually displayed
        if (slow_down):
            self.optimize_time = False
            self.use_constraint = True
            self.num_steps_planning = 50 * 4
        idx_to_be_moved = idx_to_be_moved.astype(int)
        self.idx_to_be_moved = idx_to_be_moved
        self.list_unplaced = list_unplaced
        if (initial_pose is None):
            initial_pose = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.06, 0.06]
        # enable the logging of the velocity
        self.log_velocity = True
        self.check_contacts_on_grasp = True
        self.logging_list = list_unplaced
        self.logging_arr = np.zeros((len(self.logging_list)))
        # execute the normal grasp part function
        success = self.grasp_part(pos,ori,obj_to_be_moved,initial_pose=initial_pose)
        # disable again the logging of the velocities as this is not required any more
        self.log_velocity = False
        self.check_contacts_on_grasp = False

        vel_wo_moved = copy.deepcopy(self.logging_arr)

        if (success):
            # check if the action was valid, i.e. we did not make contact with any other object and we did not move any
            # other object (both cases are undesired)
            if (self.num_contacts!=0):
                success = False
            if (success):
                # velocity is checked to make sure that during movement to grasp the block we do not bump into other ones
                if (np.max(vel_wo_moved)>1.0):
                    success = False
        return success


    # simpler grasp function
    def grasp_part(self,pos,ori=[0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254],obj_to_be_moved=None,initial_pose=None,slow_down=False):
        # boolean slow_down is useful if you want to have fast execution in all threads / simulations in which MCTS is
        # conducted but slower speed in the environment which is actually displayed
        if (slow_down):
            self.optimize_time = False
            self.use_constraint = True
            self.num_steps_planning = 50 * 4
        # start the inverse kinematics from the standard pose
        if (initial_pose is None):
            poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        else:
            poses = initial_pose
        self.ee_signal[-1] = poses[-1]
        self.ee_signal[-2] = poses[-2]
        self.q = np.asarray(poses)
        # first position to reach is 15 cm above the desired pose
        self.q = self.ik_pin(np.asarray(self.q),pos+[0,0,0.15], ori, ori_matters=False)
        # if self.q is None this corresponds to the IK having failed -> abort
        if (self.q is None):
            return False
        # otherwise - if optimize time -> directly set the robot to the desired pose ; not optimize time -> move the robot
        # to the desired pose using the simulator
        if (self.optimize_time):
            self.reset_robot(self.q)
        else:
            self.move_robot(self.q)
        # next pose is 5cm above
        self.q = self.ik_pin(np.asarray(self.q),pos+[0,0,0.05], ori)
        if (self.q is None):
            return False
        if (self.optimize_time):
            self.reset_robot(self.q)
        else:
            self.move_robot(self.q)
        # next pose is exactly at the position
        self.q = self.ik_pin(np.asarray(self.q),pos+[0,0,0.0], ori)
        if (self.q is None):
            return False
        # move the robot in any case
        self.move_robot(self.q,factor = 0.5)

        if (self.check_contacts_on_grasp):
            # check potential contacts / collisions with all other unplaced objects:
            self.num_contacts = 0
            for i in range(len(self.list_unplaced)):
                self.num_contacts += len(
                    p.getContactPoints(bodyA=self.robot_id, bodyB=self.list_unplaced[i].handle,
                                       physicsClientId=self.physicsClient))

        self.step_sim()
        self.log_velocity = False

        if self.use_constraint:
            # use constraint for grasping / making contact with the part, i.e. create ridgid link between end effector
            # and the part
            pos1, ori1 = obj_to_be_moved.get_pos_orient()

            position = self.joint_pos(np.asarray(self.q))#+[0, 0, -0.5]
            orientation_of_constraint_frame = (p.getLinkState(self.robot_id,11,physicsClientId=self.physicsClient))[1]
            self.cid = p.createConstraint(obj_to_be_moved.handle, -1, self.robot_id, 11, p.JOINT_FIXED,
                                     position-pos1, [0, 0, 0],
                                     [0, 0, 0.0],childFrameOrientation=orientation_of_constraint_frame,physicsClientId=self.physicsClient)

        if not(self.optimize_time):
            self.move_robot(self.q, factor=0.5)
            # closing gripper only done when not optimizing time - otherwise the constraint takes care of establishing
            # a connection between the gripper and the part
            self.close_gripper()
            self.ee_signal[-1] = 0.01
            self.ee_signal[-2] = 0.01

        # finally move 20 cm above the desired part,...
        self.q = self.ik_pin(np.asarray(self.q), pos + [0, 0, 0.2], ori)
        if (self.q is None):
            return False
        if (self.optimize_time):
            self.reset_robot(self.q)
        else:
            self.move_robot(self.q)

        return True

    # placing the part
    def place_part_w_table_collision_check(self, pos, p_offsets,
                   ori=[0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254],lower_obj=None, obj_to_be_moded=None, object_list=None, list_unplaced=None,idx_to_be_moved=None, grasp_bias=None,slow_down=False):
        # again same reasoning as in the functions above regarding MCTS
        if (slow_down):
            self.optimize_time = False
            self.use_constraint = True
            self.num_steps_planning = 50 * 4

        self.check_table_collision = True
        self.let_part_fall_till_rest = True
        self.idx_to_be_moved = idx_to_be_moved.astype(int)
        self.list_unplaced = list_unplaced
        # call place part function
        success = self.place_part(pos,p_offsets, ori,lower_obj,obj_to_be_moded, object_list,grasp_bias=grasp_bias)
        if (success):
            if (self.num_contacts!=0):
                success = False

        return success

    # function to perform the placing of the parts
    def place_part(self,pos,p_offsets, ori=[0.7071754336357117, -0.7070379257202148, 0.0003535877913236618, -0.00035351901897229254],lower_obj=None,obj_to_be_moded=None, object_list = None, grasp_bias=None, slow_down=False):
        if (slow_down):
            self.optimize_time = False
            self.use_constraint = True
            self.num_steps_planning = 50 * 4
        if not(grasp_bias is None):
            # this bias warm starts the inverse kinematics solving - this is needed to ensure a good solution of the IK
            # as otherwise the solver cannot find a solution to the commanded grasp pose
            self.q = np.asarray(grasp_bias)

        self.integral = np.zeros((9))
        # the list p_offsets consists of waypoints that determine how the part is being placed
        for i in range(len(p_offsets)):
            pose = (pos + p_offsets[i], ori)
            self.q = self.ik_pin(np.asarray(self.q), pose[0], pose[1])
            # again if IK fails to find a solution -> exit immediately
            if (self.q is None):
                return False
            if (i<=2):
                if not(self.optimize_time):
                    if (i == 0):
                        # first do more smooth movemement
                        self.move_robot_full(self.q, obj_to_be_moded, pos + p_offsets[i], soft=True)

                self.move_robot_full(self.q, obj_to_be_moded, pos + p_offsets[i],add_factor=2.0)
                if (i==1):
                    if (self.check_table_collision):
                        # check for collisions with table now (needed when placing larger parts)
                        self.num_contacts = 0
                        for i in range(len(self.idx_to_be_moved)):
                            self.num_contacts += len(p.getContactPoints(bodyA=self.table_id, bodyB=self.list_unplaced[
                                self.idx_to_be_moved[i]].handle, physicsClientId=self.physicsClient))
            # on the 4th waypoint, i.e. i=0,...,3 we actually remove the constraint
            elif (i == 3):
                self.move_robot_full(self.q, obj_to_be_moded, pos + p_offsets[i],add_factor=4.0)
                if (self.use_constraint):
                    p.removeConstraint(self.cid, physicsClientId=self.physicsClient)
            else:
                if (self.optimize_time):
                    None
                else:
                    self.move_robot_full(self.q,obj_to_be_moded,pos + p_offsets[i])

        # then we open the gripper
        success = self.open_gripper(object_list)
        if not(self.optimize_time):
            self.ee_signal[-1] = 0.08
            self.ee_signal[-2] = 0.08
        if (not success):
            return  False

        # next we reset the robot into some reasonable position from which we then attempt the grasping of the next part
        # this is done by following 3 waypoints
        # waypoint 1
        pose = (pos + [*p_offsets[0][:2],0] + [0,0,0.1], ori)
        self.q = self.ik_pin(np.asarray(self.q), pose[0], pose[1])
        if (self.q is None):
            return False
        if (self.optimize_time):
            self.reset_robot(self.q)
        else:
            self.move_robot(self.q)
        # waypoint 2
        pose = (pos + [*p_offsets[0][:2],0] + [0,0,0.25], ori)
        self.q = self.ik_pin(np.asarray(self.q), pose[0], pose[1])
        if (self.q is None):
            return False
        if (self.optimize_time):
            self.reset_robot(self.q)
        else:
            self.move_robot(self.q)
        # waypoint 3
        pose = (pos + [*p_offsets[0][:2],0] + [0.0,-0.15,0.25], ori)
        self.q = self.ik_pin(np.asarray(self.q), pose[0], pose[1], ori_matters=False)
        if (self.q is None):
            return False
        if (self.optimize_time):
            self.reset_robot(self.q)
        else:
            self.move_robot(self.q)
        # if none of the things failed before - we finally return True -> successfull placing of the part,...
        return True

    # function that performs the opening of the gripper
    def open_gripper(self,object_list):
        p.setJointMotorControl2(bodyIndex=self.robot_id,
                                jointIndex=9,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0.08,
                                targetVelocity=0,
                                force=500,
                                physicsClientId=self.physicsClient)
        p.setJointMotorControl2(bodyIndex=self.robot_id,
                                jointIndex=10,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0.08,
                                targetVelocity=0,
                                force=500,
                                physicsClientId=self.physicsClient)

        # if before opening of the gripper we bumped into any other object -> opening was unsuccesfull,...
        num_contacts = 0
        for i in range(len(object_list)-1):
            num_contacts += len(
                p.getContactPoints(bodyA=self.robot_id, bodyB=object_list[i].handle,
                                   physicsClientId=self.physicsClient))

        if (num_contacts>0):
            return False

        # actually open the gripper as we also step the simulation here,...
        acc_vel_lin1 = np.zeros((len(object_list)-1))
        for j in range(1*self.num_steps_planning):
            self.step()
            for i in range(len(object_list)-1):
                curr_lin, curr_ang = object_list[i].get_vel()
                acc_vel_lin1[i] += np.sum(np.abs(curr_lin))

        # if after opening of the gripper we are in contact with any other object -> opening was unsuccesfull,...
        for i in range(len(object_list)-1):
            num_contacts += len(
                p.getContactPoints(bodyA=self.robot_id, bodyB=object_list[i].handle,
                                   physicsClientId=self.physicsClient))

        success = True
        if (num_contacts>0):
            return False
        # depending on how many steps we did the stepping potentially adjust the threshold
        threshold = 0.75
        if not(self.optimize_time):
            threshold = threshold*3
        # if threshold is exceeded - parts moved too much - invalid action detected
        if (np.max(acc_vel_lin1)>threshold):
            return False

        # if this variable is set we let the part fall until it has settled or a time threshold has been exceeded,...
        if (self.let_part_fall_till_rest):
            if (success):
                counter = 0
                threshold = 200
                while(np.abs(object_list[-1].get_vel()[0][2])>0.1 and counter<threshold):
                    self.step()
                    counter += 1

        return success

    # function performing the closing of the gripper
    def close_gripper(self):
        p.setJointMotorControl2(bodyIndex=self.robot_id,
                                jointIndex=9,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0.01,
                                targetVelocity=0,
                                force=500,
                                physicsClientId=self.physicsClient)
        p.setJointMotorControl2(bodyIndex=self.robot_id,
                                jointIndex=10,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0.01,
                                targetVelocity=0,
                                force=500,
                                physicsClientId=self.physicsClient)

        for j in range(self.num_steps_planning):
            self.step()

    # function is needed to check the initial stability if environment is populated randomly
    def check_initial_stability(self,object_list,object_list_unplaced,use_robo=True):
        acc_vel_lin1 = np.zeros((len(object_list)))
        acc_vel_lin2 = np.zeros((len(object_list_unplaced)))
        for j in range(10):
            self.step()
            for i in range(len(object_list)):
                curr_lin, curr_ang = object_list[i].get_vel()
                acc_vel_lin1[i] += np.sum(np.abs(curr_lin))
            for i in range(len(object_list_unplaced)):
                curr_lin, curr_ang = object_list_unplaced[i].get_vel()
                acc_vel_lin2[i] += np.sum(np.abs(curr_lin))
        if (use_robo):
            self.reset_robot()

        if (len(object_list)!=0 and len(object_list_unplaced)!=0):
            if (np.max(acc_vel_lin1)>1.25 or np.max(acc_vel_lin2)>1.25):
                success  = False
            else:
                success = True
        elif (len(object_list)==0 and len(object_list_unplaced)!=0):
            if (np.max(acc_vel_lin2)>1.25):
                success  = False
            else:
                success = True
        elif (len(object_list)!=0 and len(object_list_unplaced)==0):
            if (np.max(acc_vel_lin1)>1.25):
                success  = False
            else:
                success = True

        return success

    # special function for stepping the simulation for 10 timesteps
    def step_sim(self):
        for j in range(10):
            self.step()

    # more advanced function to move the robot based on a desired task space position to be used
    def move_robot_full(self, next_pos,obj_to_be_moded,target, add_factor=1.0, soft=False):
        use_int = False
        if (self.optimize_time):
            num_steps = int(self.num_steps_planning*0.5*add_factor)
        else:
            num_steps = int(self.num_steps_planning)

        for j in range(int(num_steps)):
            pos_h, ori_abc = obj_to_be_moded.get_pos_orient()
            poses, velocities = self.get_robot_info()
            error = (target-pos_h)
            factor = 0.5
            if (soft):
                factor = 0.15
            if (np.max(np.abs(error))<0.01 or use_int):
                use_int = True
                vel = self.ik_residual(np.asarray(poses),error)
                self.integral -= vel
            else:
                self.integral = np.zeros((9))

            for i in range(7):
                poses, velocities = self.get_robot_info()
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=poses[i] + factor*(next_pos[i]-poses[i]) + 0.0075*self.integral[i],
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self.physicsClient)
            # if not optimize the time -> also make sure to control the gripper
            if not(self.optimize_time):
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=9,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.ee_signal[-2],
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self.physicsClient)
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=10,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.ee_signal[-1],
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self.physicsClient)

            self.step()

    # simple function to move the robot to reach a desired joint configuration,...
    def move_robot(self, next_pos,factor=1.0):
        for j in range(int(self.num_steps_planning*factor)):
            for i in range(7):
                poses, velocities = self.get_robot_info()
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=poses[i] + 0.5*(next_pos[i]-poses[i]),
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self.physicsClient)

            if not (self.optimize_time):
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=9,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.ee_signal[-2],
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self.physicsClient)
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=10,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.ee_signal[-1],
                                        targetVelocity=0,
                                        force=500,
                                        physicsClientId=self.physicsClient)

            self.step()
            if (self.log_velocity):
                self.do_log_velocity()

    # return the residual of the IK
    def ik_residual(self,q, error):
        damp = 1e-12
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacement(self.model, self.data, self.frameId)
        J = pinocchio.computeFrameJacobian(self.model, self.data, q, self.frameId, pinocchio.ReferenceFrame.WORLD)[:3]
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), error[:3]))
        return v

    # returns the joint position of our "grasping frame" - this functionality is mainly needed for creating the constraint
    def joint_pos(self,q):
        if (self.data is None):
            self.data = self.model.createData()
        pinocchio.forwardKinematics(self.model, self.data, q)
        pinocchio.updateFramePlacement(self.model, self.data, self.frameId)
        return (self.data.oMf[self.frameId].translation)

    # implementation of inverse kinematics with pinocchio
    def ik_pin(self,q, pos, ori, ori_matters=True):
        # only needed as data cannot be pickled atm
        if (self.data is None):
            self.data = self.model.createData()
        # add offset because of the coordinate system of the environment
        pos = pos + [0.5,0,0]

        xyzquat = [*pos, *ori]
        oMdes = pinocchio.XYZQUATToSE3(xyzquat)

        eps = 1e-6
        IT_MAX = 1000
        DT = 1e-1
        damp = 1e-12

        i = 0
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacement(self.model, self.data, self.frameId)
            dMi = oMdes.actInv(self.data.oMf[self.frameId])
            err = pinocchio.log(dMi).vector
            # depending if orientation should be taken into account or not, we maybe set the error to 0
            if (not ori_matters):
                err[3:] = 0.0
            if norm(err) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break

            if (ori_matters):
                J = pinocchio.computeFrameJacobian(self.model, self.data, q, self.frameId,pinocchio.ReferenceFrame.LOCAL)
                v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
            else:
                J = pinocchio.computeFrameJacobian(self.model, self.data, q, self.frameId, pinocchio.ReferenceFrame.LOCAL)[:3]
                v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(3), err[:3]))
            q = pinocchio.integrate(self.model, q, v * DT)
            i += 1

        if success:
            if (self.verbose):
                print("Convergence achieved!")
            return q.flatten().tolist()

        else:
            print ("IK FAILED!!!")
            if (self.verbose):
                print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
            return None
