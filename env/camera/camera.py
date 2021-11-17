# This class should serve as a modular definition of cameras

import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

class Camera:

    def __init__(self, cam_pos, cam_target_pos, cam_up_vector, cam_width=512, cam_height=512, fov=120, near=0.25, far=2.0, half_height=False, simId=0, static=True, related_obj=None):
        # define camera:
        self.cam_width = cam_width#512  # 128
        self.cam_height = cam_height#512  # 128
        self.fov = fov#120  # 60
        self.aspect = self.cam_height / self.cam_width
        self.near = near #0.25  # 0.02
        self.far = far #2.0  # 1
        self.cam_pos = cam_pos
        self.cam_target_pos = cam_target_pos
        self.cam_up_vector = cam_up_vector
        self.static = static
        self.physicsClient = simId

        if not(self.static):
            self.related_obj = related_obj
            self.rel_cam_pos = cam_pos
            self.rel_target_pos = cam_target_pos
            self.update_dynamic_params()
        else:
            self.related_obj = None

        self.view_matrix = None
        self.projection_matrix = None
        # initialize parameters
        self.compute_cam_params()

        self.half_height = half_height

    def compute_cam_params(self):
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.cam_pos,
            cameraTargetPosition=self.cam_target_pos,
            cameraUpVector=self.cam_up_vector,
            physicsClientId=self.physicsClient)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far, physicsClientId=self.physicsClient)

    def update_dynamic_params(self):
        part_pos, part_orn = self.related_obj.get_pos_orient()
        self.cam_pos = part_pos + self.rel_cam_pos
        self.cam_target_pos = part_pos + self.rel_target_pos
        self.compute_cam_params()

    def render(self):
        if not(self.static):
            self.update_dynamic_params()

        self.images = p.getCameraImage(self.cam_width,
                                  self.cam_height,
                                  self.view_matrix,
                                  self.projection_matrix,
                                  shadow=True,
                                  renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                  physicsClientId=self.physicsClient)

        self.rgb_opengl = np.reshape(self.images[2], (self.cam_height, self.cam_width, 4)) * 1. / 255.
        # depth_buffer_opengl = np.reshape(self.images[3], [self.cam_width, self.cam_height])
        depth_buffer_opengl = self.images[3]

        self.depth_opengl = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_opengl)
        # # apply additional "rectification":
        # self.depth_opengl = np.multiply(self.depth_opengl,self.depth_rectification_mat)
        self.seg_opengl = np.reshape(self.images[4], [self.cam_width, self.cam_height]) * 1. / 255.

        self.remove_gnd_rgb = self.rgb_opengl[:int(self.cam_height/2),:,:]
        self.remove_gnd_depth = self.depth_opengl[:int(self.cam_height/2),:]

    def get_rgb(self):
        if (self.half_height):
           return self.remove_gnd_rgb
        return self.rgb_opengl

    def get_rgb_wo_gnd(self):
        return self.remove_gnd_rgb

    def get_depth(self, filter=None):
        if (filter is None):
            if (self.half_height):
               return self.remove_gnd_depth
            return self.depth_opengl
        else:
            self.depth_opengl[self.depth_opengl<filter[0]] = 0
            self.depth_opengl[self.depth_opengl>filter[1]] = 0
            self.depth_opengl[self.depth_opengl!=0] = 1
            # u,v = self.map_3d_to_pix([-0.025,-0.025,0.01])
            # self.depth_opengl[int(v),int(u)] = 1.0
            # self.depth_opengl[79,50] = 1.0
            return self.depth_opengl

    def get_depth_wo_gnd(self):
        return self.remove_gnd_depth

    def get_segment(self):
        return self.seg_opengl

    def show_multiple_rgb(self,images,name):
        '''
        Assumes: images and names passed as list -> visualizes all of them together
        :param images:
        :param name:
        :return: shows an image,...
        '''

        for i in range(len(images)):
            plt.subplot(len(images), 1, (i+1))
            plt.imshow(images[i])
            plt.title(name[i])

        plt.show()

    def show_multiple_depth(self,images,name):
        '''
        Assumes: images and names passed as list -> visualizes all of them together
        :param images:
        :param name:
        :return: shows an image,...
        '''

        for i in range(len(images)):
            plt.subplot(len(images), 1, (i + 1))
            plt.imshow(images[i],cmap='gray', vmin=0, vmax=1)
            plt.title(name[i])

        plt.show()
        # plt.show(block=False)
        # plt.pause(-1)
        # plt.close()


    def show_rgb(self):
        plt.imshow(self.rgb_opengl)
        plt.title('RGB OpenGL3')
        plt.show()

    def show_rgb_wo_gnd(self):
        plt.imshow(self.remove_gnd_rgb)
        plt.title('RGB OpenGL3')
        plt.show()

    def show_depth(self):
        plt.imshow(self.depth_opengl, cmap='gray', vmin=0, vmax=1)
        plt.title('Depth OpenGL3')
        plt.show()

    def show_depth_wo_gnd(self):
        plt.imshow(self.remove_gnd_depth, cmap='gray', vmin=0, vmax=1)
        plt.title('Depth OpenGL3')
        plt.show()

    def show_segment(self):
        plt.imshow(self.seg_opengl)
        plt.title('Seg OpenGL3')
        plt.show()

    def map_3d_to_pix(self, point3d):
        point = np.zeros((4))
        point[:3] = np.asarray(point3d)[:]
        point[3] = 1.0
        # print(point)

        view_mat = np.asarray(self.view_matrix).reshape([4, 4], order='F')
        interm_point = (np.matmul(view_mat, point))

        proj_mat = np.asarray(self.projection_matrix).reshape([4, 4], order='F')

        final_orig = (np.matmul(proj_mat, interm_point))

        final_y = self.cam_height - ((final_orig[1] / final_orig[3] + 1) / 2) * self.cam_height
        final_x =  ((final_orig[0] / final_orig[3] + 1) / 2) * self.cam_width

        # round intelligently:
        if (final_y>self.cam_height/2):
            final_y = np.floor(final_y)
        else:
            final_y = np.ceil(final_y)

        if (final_x>self.cam_width/2):
            final_x = np.floor(final_x)
        else:
            final_x = np.ceil(final_x)

        return final_x, final_y
