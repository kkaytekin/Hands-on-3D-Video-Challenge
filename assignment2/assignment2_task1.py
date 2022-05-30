import pyrender, trimesh, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def ros_pose_to_mat_pose(ros_pose):
    mat_pose = np.identity(4)
    rot = R.from_quat(ros_pose[3:]).as_dcm()
    trans = ros_pose[:3]

    mat_pose[:3, :3] = rot
    mat_pose[:3, 3] = trans

    ee_to_left = mat_pose
    return ee_to_left

def invert_transform(transform):
    out = np.identity(4)
    rot_transpose = transform[:3,:3].transpose()
    trans = transform[:3,3]
    out[:3,:3] = rot_transpose
    out[:3, 3] = -np.matmul(rot_transpose,trans)
    return out

def main():

    ############################################################################
    #####        loading necessary files. do not modify anything           #####
    ############################################################################

    ## loading obj pose
    obj_name =  "cutlery_spoon_3"
    pose = np.loadtxt(os.path.join("./obj_meshes",obj_name,obj_name+".txt")) #T_wc

    ## read image
    rgb = plt.imread("000000.png")

    ############################################################################
    #####                  Your part starts from here !                    #####
    ############################################################################

    ### Step 1. ###
    # Find out the orientation of the object pose by inspecting the camera pose and object pose

    # Step 1-1
    # Lets first obtain relative pose between the camera and the object (obj_to_rgb)
    # we will use the "white spatula spoon" as the obj (located at the "lower left" part of the image center)
    # you can form relative pose graph between "obj", "RB", "EE", "rgb" by using "obj_to_RB", "EE_to_RB", "rgb_to_EE"
    # to obtain the relative pose "obj_to_rgb"

    rgb_to_EE = ros_pose_to_mat_pose(np.loadtxt("pol_to_ee.txt"))
    EE_to_RB = np.loadtxt("000000.txt")
    obj_to_RB = np.array(pose)

    obj_to_rgb =  invert_transform(rgb_to_EE) @ invert_transform(EE_to_RB) @ obj_to_RB

    # Step 1-2
    # Lets see the translation part of the object from the image center from the "obj_to_rgb" and compare with the
    # object seen from the image. Check the offset of the object from the center of the image and guess the
    # coordinate system

    print(obj_to_rgb[:3,3])

    plt.figure()
    plt.imshow(rgb)
    plt.show()
    
    # Answer: The printed statement is the 3D position of the spoon observed from the camera reference frame. We know that
    # [0,0,1] corresponds to camera center. Also. z-axis points towards the image, x-axis towards right and y-axis towards down.
    # The coordinates we read are in consistency with the image. Spoon is located at the upper-left side of the image center.

if __name__ == "__main__":
    main()






