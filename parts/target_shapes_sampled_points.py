import sys
sys.path.append("../")
import pybullet

import numpy as np

# Sample the target shape to be built now through defining the corner points

class Blocks:

    def __init__(self):
        pass

    def get_random_target_shape(self):
        pass

    def visualize_shape(self):
        pass


def isRight(a, b, c):
    # adapted from here: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    # this function was needed for the image,... and tells us whether a point is right or (potentially left) of the curve
    '''
    Function / curve is defined from a-b and c is point to be checked,..
    '''
    return ((b[0] - a[0])*(c[:,:,1] - a[1]) - (b[1] - a[1])*(c[:,:,0] - a[0])) < 0

def isLeftPoint(a, b, c,idx):
    # adapted from here: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    '''
    Function / curve is defined from a-b and c is point to be checked,..
    '''
    return ((b[idx[0]] - a[idx[0]])*(c[idx[1]] - a[idx[1]]) - (b[idx[1]] - a[idx[1]])*(c[idx[0]] - a[idx[0]])) > 0

def isRightPoint(a, b, c,idx):
    # adapted from here: https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line
    '''
    Function / curve is defined from a-b and c is point to be checked,..
    '''
    return ((b[idx[0]] - a[idx[0]])*(c[idx[1]] - a[idx[1]]) - (b[idx[1]] - a[idx[1]])*(c[idx[0]] - a[idx[0]])) < 0


class TargetsBlockSquare(Blocks):
    # this function samples a triangular target shape consosting of 4 corner points
    def __init__(self,camera,rotation=np.eye(3),translation=np.asarray([0,0,0]),fliplr=False,width=3,height=3):
        self.camera_handle = camera
        self.blocks_dimensions = [0.05 / 2, 0.05 / 2, 0.05 / 2]

        self.rotation = rotation
        self.translation = translation
        self.fliplr = fliplr

        self.target_list = []
        self.eff_width = (width-1)/2 # substract 1 for the center element as the target shape should be centered
        self.eff_height = height

        self.get_random_target_shape()

    def get_random_target_shape(self, default=[]):
        # sample a random target shape
        eff_width = self.eff_width
        if (default==[]):
            left_bottom = np.asarray([np.random.choice([0.5,0.9])*(-1+-2*(eff_width)),-1,0])*self.blocks_dimensions[0]
            right_bottom = np.asarray([np.random.choice([0.5,0.9])*(1+2*(eff_width)),-1,0])*self.blocks_dimensions[0]
            right_top = np.asarray([(2*np.random.choice([0.5,0.9])-1)*(1 + 2 * (eff_width)), -1, np.random.choice([0.5,0.9])*(2*self.eff_height)]) * self.blocks_dimensions[0]
            self.real_world_point_list = []
            # the shape consists of 3 corner points. To define the entire shape -> we close the loop
            self.real_world_point_list.extend((left_bottom,right_top,right_bottom,left_bottom))
        else:
            # if we are already given the points we have to apply the transformation to transform them back into the standard
            # frmae
            self.real_world_point_list = default
            self.real_world_point_list = np.transpose(np.matmul(np.linalg.inv(self.rotation),
                                                                np.add(np.transpose(self.real_world_point_list),
                                                                       -self.translation.reshape(3, -1))))
            self.real_world_point_list = list(self.real_world_point_list)
            self.real_world_point_list.append(self.real_world_point_list[0])
        # pix list is the pixel coordinates of the points
        self.pix_list = []
        for i in range(len(self.real_world_point_list)):
            u, v = self.camera_handle.map_3d_to_pix(self.real_world_point_list[i])
            self.pix_list.append([u,-v]) #treat the y-axis negatively to account for the reversed logic
        # now create a masking array by projecting the 3D points into the camera frame and constructing the shape
        width = self.camera_handle.cam_width
        height = self.camera_handle.cam_height
        overall_mask = np.zeros((width,height,2))
        overall_mask[:,:,0] = np.transpose(np.tile(np.linspace(0,width, width, endpoint=False), (height,1)))
        overall_mask[:,:,1] = -(np.tile(np.linspace(0,height, height, endpoint=False), (width,1)))

        masking_arrs = []
        for i in range(len(self.pix_list)-1):
            masking_arrs.append((isRight(self.pix_list[i],self.pix_list[i+1],overall_mask)))

        resulting_mask = masking_arrs[0]
        for i in range(len(masking_arrs) - 1):
            resulting_mask = np.logical_and(resulting_mask,masking_arrs[i+1])

        # has to be switched as x,y is swapped in pixels,..
        resulting_mask = np.transpose(resulting_mask)
        self.target_shape = resulting_mask
        if (self.fliplr):
            self.target_shape = np.fliplr(self.target_shape)

        self.real_world_point_list = np.transpose(np.add(np.matmul(self.rotation,np.transpose(self.real_world_point_list)),self.translation.reshape(3,-1)))
        self.real_world_point_list = list(self.real_world_point_list)

        return self.target_shape, self.real_world_point_list[:-1]

    def visualize_shape(self, physicsClient):
        # function to visualize the target shape through debug lines
        num_elements = len(self.real_world_point_list)
        for i in range(num_elements-1):
            pybullet.addUserDebugLine([self.real_world_point_list[i][0],self.real_world_point_list[i][1],self.real_world_point_list[i][2]],
                                      [self.real_world_point_list[i+1][0],self.real_world_point_list[i+1][1],self.real_world_point_list[i+1][2]],
                                      lineColorRGB=[1,1,0],lineWidth=5,lifeTime=2,physicsClientId=physicsClient)


class TargetsBlockSquareUp5(TargetsBlockSquare):
    # compute a target shape of up to 5 elements
    def __init__(self,camera,rotation=np.eye(3),translation=np.asarray([0,0,0]),fliplr=False, width=3,height=3):
        self.camera_handle = camera
        self.blocks_dimensions = [0.05 / 2, 0.05 / 2, 0.05 / 2]

        self.rotation = rotation
        self.translation = translation
        self.fliplr = fliplr

        self.target_list = []

        self.eff_width = (width-1)/2
        self.eff_height = height

        self.width_lb = (3-1)/2
        self.height_lb = 3
        self.use_lb = False

        self.get_random_target_shape()

    def get_random_target_shape(self,default=[]):
        eff_width = self.eff_width
        # sample randomly 3,4 or 5 target points which define the shape
        if (default == []):
            num_blocks = np.random.choice([3,4,5])

            if (num_blocks==3):
                # we also potentially define a bound which allows us to ensure that we only sample novel shapes by
                # enabling the lower bound
                left_bottom_bound = np.asarray([1.0*(-1+-2*(self.width_lb)),-1,0])*self.blocks_dimensions[0]
                right_bottom_bound = np.asarray([1.0 * (1 + 2 * (self.width_lb)), -1, 0]) * self.blocks_dimensions[
                    0]
                right_top_bound_up = np.asarray([(2*1.0-1)*(1 + 2 * (self.width_lb)), -1, 1.0*(2*self.height_lb)]) * self.blocks_dimensions[0]
                right_top_bound_down = np.asarray([(2*0.0-1)*(1 + 2 * (self.width_lb)), -1, 1.0*(2*self.height_lb)]) * self.blocks_dimensions[0]

                while True:
                    left_bottom = np.asarray([np.random.sample()*(-1+-2*(eff_width)),-1,0])*self.blocks_dimensions[0]
                    right_bottom = np.asarray([np.random.sample()*(1+2*(eff_width)),-1,0])*self.blocks_dimensions[0]
                    right_top = np.asarray([(2*np.random.sample()-1)*(1 + 2 * (eff_width)), -1, np.random.sample()*(2*self.eff_height)]) * self.blocks_dimensions[0]
                    # only if all sampled points are outside the decider is true
                    decider = (left_bottom<left_bottom_bound).any() or (right_bottom>right_bottom_bound).any() or (right_top[:1]<right_top_bound_down[:1]).any() or (right_top>right_top_bound_up).any()
                    if (self.use_lb):
                        if (decider):
                            break
                    else:
                        break

                self.real_world_point_list = []
                self.real_world_point_list.extend((left_bottom,right_top,right_bottom,left_bottom))
                self.middle_list_x = []
                self.middle_list_x.extend((False, False, False, False))
            if (num_blocks==4):
                left_bottom_bound = np.asarray([1.0*(-1+-2*(self.width_lb)),-1,0])*self.blocks_dimensions[0]
                right_bottom_bound = np.asarray([1.0 * (1 + 2 * (self.width_lb)), -1, 0]) * self.blocks_dimensions[
                    0]
                left_top_bound = np.asarray([1.0 * (-1 + -2 * (self.width_lb)), -1,
                                       np.random.sample() * (2 * self.height_lb)]) * self.blocks_dimensions[0]

                right_top_bound = np.asarray([(1.0)*(1 + 2 * (self.width_lb)), -1, 1.0*(2*self.height_lb)]) * self.blocks_dimensions[0]

                while True:
                    left_bottom = np.asarray([np.random.sample()*(-1+-2*(eff_width)),-1,0])*self.blocks_dimensions[0]
                    right_bottom = np.asarray([np.random.sample()*(1+2*(eff_width)),-1,0])*self.blocks_dimensions[0]
                    left_top = np.asarray([np.random.sample() * (-1 + -2 * (eff_width)), -1,
                                            np.random.sample() * (2 * self.eff_height)]) * self.blocks_dimensions[0]
                    right_top = np.asarray([(np.random.sample())*(1 + 2 * (eff_width)), -1, np.random.sample()*(2*self.eff_height)]) * self.blocks_dimensions[0]

                    decider = (left_bottom<left_bottom_bound).any() or (right_bottom>right_bottom_bound).any() or \
                              (left_top[0]<left_top_bound[0]).any() or (left_top[2]>left_top_bound[2]).any() or \
                              (right_top>right_top_bound).any()
                    if (self.use_lb):
                        if (decider):
                            break
                    else:
                        break
                self.real_world_point_list = []
                self.real_world_point_list.extend((left_bottom,left_top,right_top,right_bottom,left_bottom))
                self.middle_list_x = []
                self.middle_list_x.extend((False, False, False, False, False))
            if (num_blocks==5):
                left_bottom_bound = np.asarray([1.0 * (-1 + -2 * (self.width_lb)), -1, 0]) * self.blocks_dimensions[0]
                right_bottom_bound = np.asarray([1.0 * (1 + 2 * (self.width_lb)), -1, 0]) * self.blocks_dimensions[
                    0]
                left_top_bound = np.asarray([1.0 * (-1 + -2 * (self.width_lb)), -1,
                                             np.random.sample() * (2 * self.height_lb)]) * self.blocks_dimensions[0]

                right_top_bound = np.asarray([(1.0) * (1 + 2 * (self.width_lb)), -1, 1.0 * (2 * self.height_lb)]) * \
                                  self.blocks_dimensions[0]
                middle_top_bound = np.asarray(
                    [(np.random.sample()) * (1 + 2 * (eff_width)), -1, 1.0 * (2 * self.height_lb)]) * \
                             self.blocks_dimensions[0]

                while True:
                    left_bottom = np.asarray([np.random.sample()*(-1+-2*(eff_width)),-1,0])*self.blocks_dimensions[0]
                    right_bottom = np.asarray([np.random.sample()*(1+2*(eff_width)),-1,0])*self.blocks_dimensions[0]
                    left_top = np.asarray([np.random.sample() * (-1 + -2 * (eff_width)), -1,
                                            np.random.sample() * (2 * self.eff_height)]) * self.blocks_dimensions[0]
                    right_top = np.asarray([(np.random.sample())*(1 + 2 * (eff_width)), -1, np.random.sample()*(2*self.eff_height)]) * self.blocks_dimensions[0]
                    while True:
                        middle_top = np.asarray([(np.random.sample())*(1 + 2 * (eff_width)), -1, np.random.sample()*(2*self.eff_height)]) * self.blocks_dimensions[0]
                        middle_top[0] = np.random.sample()*(right_top[0]-left_top[0]) + left_top[0]
                        if (isLeftPoint(right_bottom,right_top,middle_top,[0,2]) and isRightPoint(left_bottom,left_top,middle_top,[0,2])): #check x,z coords,...
                            break
                    decider = (left_bottom<left_bottom_bound).any() or (right_bottom>right_bottom_bound).any() or \
                              (left_top[0]<left_top_bound[0]).any() or (left_top[2]>left_top_bound[2]).any() or \
                              (right_top>right_top_bound).any() or (middle_top[2]>middle_top_bound[2]).any()
                    if (self.use_lb):
                        if (decider):
                            break
                    else:
                        break

                self.real_world_point_list = []
                self.real_world_point_list.extend((left_bottom,left_top,middle_top,right_top,right_bottom,left_bottom))
                self.middle_list_x = []
                self.middle_list_x.extend((False, False, True, False, False, False, False))
        else:
            self.real_world_point_list = default
            self.real_world_point_list = np.transpose(np.matmul(np.linalg.inv(self.rotation),np.add(np.transpose(self.real_world_point_list),-self.translation.reshape(3, -1))))
            self.real_world_point_list = list(self.real_world_point_list)
            if (len(self.real_world_point_list)==3):
                self.middle_list_x =[]
                self.middle_list_x.extend((False, False, False, False))
            elif (len(self.real_world_point_list)==4):
                self.middle_list_x = []
                self.middle_list_x.extend((False, False, False, False, False))
            elif (len(self.real_world_point_list)==5):
                self.middle_list_x = []
                self.middle_list_x.extend((False, False, True, False, False, False, False))
            else:
                print ("INVALID LIST SIZE")

            self.real_world_point_list.append(self.real_world_point_list[0])
        # now again compute the masking array
        self.pix_list = []
        for i in range(len(self.real_world_point_list)):
            u, v = self.camera_handle.map_3d_to_pix(self.real_world_point_list[i])
            self.pix_list.append([u,-v]) #treat the y-axis negatively to account for the reversed logic

        width = self.camera_handle.cam_width
        height = self.camera_handle.cam_height
        overall_mask = np.zeros((width,height,2))
        overall_mask[:,:,0] = np.transpose(np.tile(np.linspace(0,width, width, endpoint=False), (height,1)))
        overall_mask[:,:,1] = -(np.tile(np.linspace(0,height, height, endpoint=False), (width,1)))

        masking_arrs = []
        for i in range(len(self.pix_list)-1):
            masking_arrs.append((isRight(self.pix_list[i],self.pix_list[i+1],overall_mask)))
            # this filtering is needed to make sure that the shape is represented correctly
            if (self.middle_list_x[i+1]==True):
                masking_arrs[-1][int(self.pix_list[i+1][0]):,:] = 1
            if (self.middle_list_x[i]==True):
                masking_arrs[-1][:int(self.pix_list[i][0]),:] = 1

        resulting_mask = masking_arrs[0]
        for i in range(len(masking_arrs) - 1):
            resulting_mask = np.logical_and(resulting_mask,masking_arrs[i+1])

        # has to be switched as x,y is swapped in pixels,..
        resulting_mask = np.transpose(resulting_mask)
        self.target_shape = resulting_mask

        if (self.fliplr):
            self.target_shape = np.fliplr(self.target_shape)

        self.real_world_point_list = np.transpose(np.add(np.matmul(self.rotation,np.transpose(self.real_world_point_list)),self.translation.reshape(3,-1)))
        self.real_world_point_list = list(self.real_world_point_list)

        return self.target_shape, self.real_world_point_list[:-1]


class TargetsBlockSquareUp6(TargetsBlockSquare):

    def __init__(self,camera,rotation=np.eye(3),translation=np.asarray([0,0,0]),fliplr=False, width=3,height=3):

        #TODO: we also need some kind of relative transform to relate cam to structure!!!

        self.camera_handle = camera
        self.blocks_dimensions = [0.05 / 2, 0.05 / 2, 0.05 / 2]

        self.rotation = rotation
        self.translation = translation
        self.fliplr = fliplr

        self.target_list = []


        self.eff_width = (width-1)/2 # substract 1 for center element,...
        self.eff_height = height

        self.width_lb = (4-1)/2
        self.height_lb = 4
        self.use_lb = False

        self.get_random_target_shape()

    def get_random_target_shape(self,default=[]):
        eff_width = self.eff_width
        if (default == []):
            num_blocks = np.random.choice([3,4,5])
            if (np.random.sample() > 0.5):

                left_bound = -0.25*np.random.sample() + 1 #(in 0.75,1.0)
                right_bound = -0.25*np.random.sample() + 1 #(in 0.75,1.0)
                portion_clip = np.random.sample() * 0.25 + 0.25
                heeight_portion =  -0.5*np.random.sample() + 0.75


                left_bottom_bound = np.asarray([left_bound * (-1 + -2 * (self.width_lb)), -1, 0]) * self.blocks_dimensions[0]
                right_bottom_bound = np.asarray([1.0 *right_bound* (1 + 2 * (self.width_lb)), -1, 0]) * self.blocks_dimensions[
                    0]
                left_top_bound = np.asarray([1.0 * left_bound*(-1 + -2 * (self.width_lb)), -1,
                                             (2 * self.height_lb)]) * self.blocks_dimensions[0]

                left_top_bound2 = np.asarray([1.0 * portion_clip* (-1 + -2 * (self.width_lb)), -1,
                                              (2 * self.height_lb)]) * self.blocks_dimensions[0]

                left_top_bound1 = np.asarray([1.0 * portion_clip*(-1 + -2 * (self.width_lb)), -1,
                                             heeight_portion*(2 * self.height_lb)]) * self.blocks_dimensions[0]

                right_top_bound = np.asarray([(1.0) *right_bound* (1 + 2 * (self.width_lb)), -1, 1.0 * heeight_portion*(2 * self.height_lb)]) * \
                                  self.blocks_dimensions[0]

                self.real_world_point_list = []
                self.real_world_point_list.extend((left_bottom_bound, left_top_bound,left_top_bound2, left_top_bound1,right_top_bound, right_bottom_bound, left_bottom_bound))
            else:

                left_bound = -0.25 * np.random.sample() + 1  # (in 0.75,1.0)
                right_bound = -0.25 * np.random.sample() + 1  # (in 0.75,1.0)
                portion_clip = np.random.sample() * 0.25 + 0.25
                heeight_portion = -0.5 * np.random.sample() + 0.75

                left_bottom_bound = np.asarray([left_bound * (-1 + -2 * (self.width_lb)), -1, 0]) * \
                                    self.blocks_dimensions[0]
                right_bottom_bound = np.asarray([1.0 * right_bound * (1 + 2 * (self.width_lb)), -1, 0]) * \
                                     self.blocks_dimensions[
                                         0]
                left_top_bound = np.asarray([1.0 *left_bound* (-1 + -2 * (self.width_lb)), -1, heeight_portion*
                                             (2 * self.height_lb)]) * self.blocks_dimensions[0]

                left_top_bound2 = np.asarray([1.0 * -portion_clip * (-1 + -2 * (self.width_lb)), -1, heeight_portion*
                                              (2 * self.height_lb)]) *self.blocks_dimensions[0]

                left_top_bound1 = np.asarray([1.0 * -portion_clip * (-1 + -2 * (self.width_lb)), -1,
                                              (2 * self.height_lb)]) * self.blocks_dimensions[0]

                right_top_bound = np.asarray([(1.0) * right_bound * (1 + 2 * (self.width_lb)), -1,
                                              1.0 * (2 * self.height_lb)]) * \
                                  self.blocks_dimensions[0]

                self.real_world_point_list = []
                self.real_world_point_list.extend((left_bottom_bound, left_top_bound, left_top_bound2, left_top_bound1,
                                                   right_top_bound, right_bottom_bound, left_bottom_bound))

            self.middle_list_x = []
            self.middle_list_x.extend((False, False, True, True, False, False, False, False))
        else:
            self.real_world_point_list = default
            self.real_world_point_list = np.transpose(np.matmul(np.linalg.inv(self.rotation),np.add(np.transpose(self.real_world_point_list),-self.translation.reshape(3, -1))))
            self.real_world_point_list = list(self.real_world_point_list)
            self.middle_list_x = []
            self.middle_list_x.extend((False, False, True, True, False, False, False, False))
            if (len(self.real_world_point_list)==8):
                self.middle_list_x = []
                self.middle_list_x.extend((False, False, True, True, False, False, False, False))
            else:
                print ("INVALID LIST SIZE")

            self.real_world_point_list.append(self.real_world_point_list[0])

        self.pix_list = []
        for i in range(len(self.real_world_point_list)):
            u, v = self.camera_handle.map_3d_to_pix(self.real_world_point_list[i])
            self.pix_list.append([u,-v]) #treat the y-axis negatively to account for the reversed logic


        width = self.camera_handle.cam_width
        height = self.camera_handle.cam_height
        overall_mask = np.zeros((width,height,2))
        overall_mask[:,:,0] = np.transpose(np.tile(np.linspace(0,width, width, endpoint=False), (height,1)))
        overall_mask[:,:,1] = -(np.tile(np.linspace(0,height, height, endpoint=False), (width,1)))

        masking_arrs = []
        for i in range(len(self.pix_list)-1):
            masking_arrs.append((isRight(self.pix_list[i],self.pix_list[i+1],overall_mask)))
            # this filtering is needed to make sure that the shape is represented correctly
            if (self.middle_list_x[i+1]==True):
                masking_arrs[-1][int(self.pix_list[i+1][0]):,:] = 1
            if (self.middle_list_x[i]==True):
                masking_arrs[-1][:int(self.pix_list[i][0]),:] = 1

        resulting_mask = masking_arrs[0]
        for i in range(len(masking_arrs) - 1):
            resulting_mask = np.logical_and(resulting_mask,masking_arrs[i+1])

        # has to be switched as x,y is swapped in pixe;s,..
        resulting_mask = np.transpose(resulting_mask)
        self.target_shape = resulting_mask

        if (self.fliplr):
            self.target_shape = np.fliplr(self.target_shape)

        self.real_world_point_list = np.transpose(np.add(np.matmul(self.rotation,np.transpose(self.real_world_point_list)),self.translation.reshape(3,-1)))
        self.real_world_point_list = list(self.real_world_point_list)
        print (len(self.real_world_point_list))

        return self.target_shape, self.real_world_point_list[:-1]