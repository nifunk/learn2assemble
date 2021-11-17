import sys
sys.path.append("../")
from mushroom_env import fill_volume_env_3D_multiple_obj_3Dac_4

import numpy as np
import time


# Generate environment
num_parts = 30
num_samples = 100#100
grid_size = 3

env = fill_volume_env_3D_multiple_obj_3Dac_4.StackBoxesEnv3D_multiple_obj_4(num_parts,visualize=True, ensemble=True, add_connectivity=True, env_grid_size=grid_size)


# Then normal interface like standart gym
# -> here now simple handtuned ctrl for sanity checking,...
obs = env.reset()
# observation consists of x,y,z coordinate and then indicator: [1,1] -> target blocks ; [1,-1] -> placed blocks ; [-1,-1] -> unplaced blocks
# i.e. 5 dimensional observation per block
# and then the connectivity is added which is again the number of blocks
original_observation = obs
reshaped = original_observation.reshape(num_parts,-1)
num_target_blocks = 0
# this function exemplarily shows how an observation can be decomposed,....
for i in range((np.shape(reshaped)[0])):
    # print (reshaped[i])
    if (reshaped[i][3]==1 and reshaped[i][4]==1):
        print ("TARGET POSITIONS") # targets are the corner points of the desired shape!
        num_target_blocks += 1
        print (reshaped[i][0],reshaped[i][1],reshaped[i][2]) # print position x,y,z
    if (reshaped[i][3]==1 and reshaped[i][4]==-1):
        print ("PLACED BLOCK")
        print(reshaped[i][0], reshaped[i][1], reshaped[i][2])  # print position x,y,z
    if (reshaped[i][3]==-1 and reshaped[i][4]==-1):
        print ("UNPLACED BLOCK")
        print(reshaped[i][0], reshaped[i][1], reshaped[i][2])  # print position x,y,z


for i in range(num_samples):
    # action is 3 dimensional: index 0 = block to be placed by specifying the index from the observation list that is decomposed as "illustrated"
    # above,... i.e. it containes first the target elements, then the placed ones, then the unplaced blocks
    # index 1 = reference block by specifying the index from the observation list
    # index 2 = relative action, there are rotations also available but for now, define as: 0;4;8;12;16

    ac = [0+num_target_blocks+i+1,i+num_target_blocks,0] # places next available unplaced block with respect to the last placed one
    # we draw a random action (possible actions are: front/back/left/right/top)
    action = np.random.choice(4)
    ac[2] = action

    obs, rew, done, _ = env.step(ac)
    print ("Reward:", rew)
    if (done):
        input("finished, press button to start anew")
        obs = env.reset()
    time.sleep(0.1)

