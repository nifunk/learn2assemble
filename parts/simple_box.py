import sys
sys.path.append("../")
import pybullet as p
import numpy as np
import os

class Simple_box:
    # this class implements the basic functionality. In the repo almost everything is framed part / object centric, so
    # here we not only create the objects and add them to the simulation but can also retrieve their position and orientation
    def __init__(self, pos, orientation, physicsClient,filler=None, direction=None):
        # create a part
        import pathlib
        self.curr_path = str(pathlib.Path(__file__).parent.resolve())

        self.physicsClient = physicsClient
        # if filler is set to 0.01 - we create a filler element that has a width of 1cm (instead of the 5cm) blocks
        # this is needed to build bigger structures and to be in accordance with the robot currently only being capable
        # of placing objects at a distance compared to other objects
        if (filler==0.01):
            if (direction=='x'):
                self.handle = p.loadURDF(self.curr_path + "/obj/simple_box/cube_small_0_01_x.urdf",
                           physicsClientId=self.physicsClient)
            else:
                self.handle = p.loadURDF(self.curr_path + "/obj/simple_box/cube_small_0_01_y.urdf",
                           physicsClientId=self.physicsClient)
        else:
            self.handle = p.loadURDF("cube_small.urdf", physicsClientId=self.physicsClient)

        p.changeDynamics(self.handle, 0, linearDamping=10, angularDamping=10, physicsClientId=self.physicsClient)
        # store dimensions of the object
        self.dimensions = [0.05 / 2, 0.05 / 2, 0.05 / 2]
        # directly reposition:
        p.resetBasePositionAndOrientation(self.handle, np.asarray(pos) + np.asarray([0,0,self.dimensions[2]]) , np.asarray(orientation),physicsClientId=self.physicsClient)
        # those are the dimensions in the x,y, and z direction each from the center of mass, assumend to be in the center
        # of the object
        self.stacking_height = [self.dimensions[0], self.dimensions[1], self.dimensions[2]/2.0]

        # This array is in order to illustrate the directions of the edges (without the real units,...)
        self.unit_edge_dir = np.zeros((3,6))
        self.unit_edge_dir[:,0] = [1,0,0]
        self.unit_edge_dir[:,1] = [-1,0,0]
        self.unit_edge_dir[:,2] = [0,1,0]
        self.unit_edge_dir[:,3] = [0,-1,0]
        self.unit_edge_dir[:,4] = [0,0,1]
        self.unit_edge_dir[:,5] = [0,0,-1]

        # list that takes care of the children in the order pos,neg then x,y,z
        self.child_list = []
        # reasoning:
        # first entry says whether something is mounted or not
        # second points to the other object
        # third one to the constraint,...
        for i in range(6):
            self.child_list.append([False,None,None])

        # This array is in order to illustrate the directions of the edges in the actual units,...
        self.real_edge_dir = np.zeros((3,6))
        self.real_edge_dir[:,0] = np.asarray([1,0,0]) * self.dimensions[0]
        self.real_edge_dir[:,1] = np.asarray([-1,0,0]) * self.dimensions[0]
        self.real_edge_dir[:,2] = np.asarray([0,1,0]) * self.dimensions[1]
        self.real_edge_dir[:,3] = np.asarray([0,-1,0]) * self.dimensions[1]
        self.real_edge_dir[:,4] = np.asarray([0,0,1]) * self.dimensions[2]
        self.real_edge_dir[:,5] = np.asarray([0,0,-1]) * self.dimensions[2]

    def apply_force(self, force):
        boxPos, boxOrn = p.getBasePositionAndOrientation(self.handle, physicsClientId=self.physicsClient)
        p.applyExternalForce(objectUniqueId=self.handle, linkIndex=-1, forceObj=force, posObj=boxPos, flags=p.WORLD_FRAME, physicsClientId=self.physicsClient)

    def apply_velocity(self,velocity):
        p.resetBasePositionAndOrientation(self.handle, self.get_pos_orient()[0], [0,0,0,1], physicsClientId=self.physicsClient)
        p.resetBaseVelocity(self.handle, np.asarray(velocity), [0,0,0], physicsClientId=self.physicsClient)

    def get_pos_orient(self):
        boxPos, boxOrn = p.getBasePositionAndOrientation(self.handle, physicsClientId=self.physicsClient)
        return np.asarray(boxPos), np.asarray(boxOrn)

    def get_vel(self):
        linVel, angVel = p.getBaseVelocity(self.handle, physicsClientId=self.physicsClient)
        return np.asarray(linVel), np.asarray(angVel)

    def get_children(self):
        return self.child_list

    def reposition(self, position, orientation):
        p.resetBasePositionAndOrientation(self.handle,np.asarray(position),np.asarray(orientation), physicsClientId=self.physicsClient)

    def change_color(self,color,transparency=1.0):
        p.changeVisualShape(self.handle, -1, rgbaColor=[color[0], color[1], color[2], transparency], physicsClientId=self.physicsClient)

    def get_edge_dir_orient(self,custom_orient):
        rotMat = np.asarray(p.getMatrixFromQuaternion(custom_orient,physicsClientId=self.physicsClient)).reshape(3,3)
        edge_dirs = np.matmul(rotMat,self.real_edge_dir)
        return edge_dirs

    def get_edge_dir(self):
        boxPos, boxOrn = p.getBasePositionAndOrientation(self.handle,physicsClientId=self.physicsClient)
        rotMat = np.asarray(p.getMatrixFromQuaternion(boxOrn,physicsClientId=self.physicsClient)).reshape(3,3)
        edge_dirs = np.matmul(rotMat,self.real_edge_dir)
        return edge_dirs

    def show_edge_dir(self,lifeTime=1.0):
        boxPos, boxOrn = p.getBasePositionAndOrientation(self.handle,physicsClientId=self.physicsClient)
        rotMat = np.asarray(p.getMatrixFromQuaternion(boxOrn,physicsClientId=self.physicsClient)).reshape(3,3)
        edge_dirs = np.matmul(rotMat,self.unit_edge_dir)
        p.addUserDebugLine(boxPos, np.asarray(boxPos) + np.asarray(edge_dirs[:,0]), lifeTime=lifeTime, lineColorRGB=[1,0,0], physicsClientId=self.physicsClient)
        p.addUserDebugLine(boxPos, np.asarray(boxPos) + np.asarray(edge_dirs[:,2]), lifeTime=lifeTime, lineColorRGB=[0,1,0], physicsClientId=self.physicsClient)
        p.addUserDebugLine(boxPos, np.asarray(boxPos) + np.asarray(edge_dirs[:,4]), lifeTime=lifeTime, lineColorRGB=[0,0,1], physicsClientId=self.physicsClient)
        return edge_dirs

    def remove(self):
        p.removeBody(self.handle, physicsClientId=self.physicsClient)

class Simple_box_filler(Simple_box):
    # this class returns a filler box element
    def __init__(self, pos, orientation, physicsClient, dir):
        super().__init__(pos, orientation, physicsClient, filler=0.01, direction=dir)

class Simple_box_with_surrounding(Simple_box):
    # create a more complex object that consists of multiple primitive elements. Also note that this is in a scenario
    # in which we also have the robot manipulator, i.e. if parts are next to each other we have to add the spacing elements
    def __init__(self, pos, orientation, physicsClient, dimensions, obj_list_unplaced, distance):
        super().__init__(pos, orientation, physicsClient)
        self.list_additional_objects = []
        # begin checking surrounding:
        # check all vor vicinity amongst the objects
        vicinity_x = np.abs(np.abs(obj_list_unplaced[:, 0] - pos[0]) - dimensions[0]) < 0.001
        vicinity_y = np.abs(np.abs(obj_list_unplaced[:, 1] - pos[1]) - dimensions[1]) < 0.001
        vicinity_z = np.abs(np.abs(obj_list_unplaced[:, 2] - pos[2]) - dimensions[2]) < 0.001

        vicinity_x_dir = -(obj_list_unplaced[:, 0] - pos[0])
        vicinity_y_dir = -(obj_list_unplaced[:, 1] - pos[1])
        vicinity_z_dir = -(obj_list_unplaced[:, 2] - pos[2])

        # check for equality:
        equal_x = np.abs(obj_list_unplaced[:, 0] - pos[0]) < 0.001
        equal_y = np.abs(obj_list_unplaced[:, 1] - pos[1]) < 0.001
        equal_z = np.abs(obj_list_unplaced[:, 2] - pos[2]) < 0.001
        # for parts to be direct neighbors in two coordinate directions we must have equal entries
        decide_arr = equal_x.astype(int) + equal_y.astype(int) + equal_z.astype(int)
        for i in range(len(decide_arr)):
            # they might be direct neighbors
            if (decide_arr[i]==2):
                # check in which direction there might be vicinity
                if (vicinity_x[i]):
                    # if x vicinity
                    if (vicinity_x_dir[i]>0):
                        new_pos = np.asarray(pos) - np.asarray([dimensions[0]/2,0.0,0.0])
                    else:
                        new_pos = np.asarray(pos) - np.asarray([-dimensions[0] / 2, 0.0, 0.0])
                    # create the filler object
                    self.list_additional_objects.append(Simple_box_filler(new_pos,np.asarray(orientation),self.physicsClient,dir='x'))
                    # create constraint for the filler object
                    cid = p.createConstraint(self.handle, -1, self.list_additional_objects[-1].handle, -1, p.JOINT_FIXED, [0, 0, 0],
                                             self.list_additional_objects[-1].get_pos_orient()[0] - self.get_pos_orient()[0],
                                             [0, 0, 0.0], physicsClientId=self.physicsClient)
                if (vicinity_y[i]):
                    # similar as for x direction
                    if (vicinity_y_dir[i]>0):
                        new_pos = np.asarray(pos) - np.asarray([0, dimensions[1]/2,0.0])
                    else:
                        new_pos = np.asarray(pos) - np.asarray([0, -dimensions[1] / 2, 0.0])
                    self.list_additional_objects.append(Simple_box_filler(new_pos,np.asarray(orientation),self.physicsClient,dir='y'))
                    cid = p.createConstraint(self.handle, -1, self.list_additional_objects[-1].handle, -1, p.JOINT_FIXED, [0, 0, 0],
                                             self.list_additional_objects[-1].get_pos_orient()[0] - self.get_pos_orient()[0],
                                             [0, 0, 0.0], physicsClientId=self.physicsClient)

    def remove(self):
        # for more complex object, we also have to remove the filler blocks
        p.removeBody(self.handle, physicsClientId=self.physicsClient)
        for i in range(len(self.list_additional_objects)):
            p.removeBody(self.list_additional_objects[i].handle, physicsClientId=self.physicsClient)
        self.list_additional_objects = []