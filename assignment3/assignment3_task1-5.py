import matplotlib.pyplot as plt
import numpy as np
import os, argparse, cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
import cv2

import pyrender,trimesh

def ros_to_mat(ros_pose):
    ee_to_left_mat = np.identity(4)
    rot = R.from_quat(ros_pose[3:]).as_dcm()
    trans = ros_pose[:3]

    ee_to_left_mat[:3, :3] = rot
    ee_to_left_mat[:3, 3] = trans

    ee_to_left = ee_to_left_mat
    return ee_to_left

def invert_transform(transform):
    out = np.identity(4)
    rot_transpose = transform[:3,:3].transpose()
    trans = transform[:3,3]
    out[:3,:3] = rot_transpose
    out[:3, 3] = -np.matmul(rot_transpose,trans)
    return out

def deproj(depth,T_relative,K_sc):

    # in: depth image of selected depth camera
    # in: transformation from depth to rgb frame
    # in: intrinsics matrix of the selected depth camera

    # Creating meshgrid
    # height, width = (480, 848) = depth.shape
    y_sc, x_sc = depth.shape

    y = np.arange(y_sc)
    x = np.arange(x_sc)

    x_grid, y_grid = np.meshgrid(x, y)
    homo = np.ones_like(x_grid)

    # homogeneous meshgrid of shape [w,h,3]
    # --> sc_mesh_homo[x,y] =  (x,y,1)
    sc_mesh_homo = np.concatenate((x_grid[:,:,None],y_grid[:,:,None],homo[:,:,None]),axis=-1)
    ##### Step 2-1
    # flatten and transpose the grid to make shape [3,x*y] to make matrix multiplication easier
    sc_mesh_homo = sc_mesh_homo.reshape((x_sc*y_sc,3))
    sc_mesh_homo = sc_mesh_homo.transpose()
    #sc_mesh_homo[[0,1],:] = sc_mesh_homo[[1,0],:]

    ##### Step 2-2
    # flatten the depth into the same shape and transpose ([1,x*y])
    depth_flattened = depth.reshape((x_sc*y_sc,1))
    depth_flattened = depth_flattened.T

    ##### Step 2-3
    # deproject the meshgrid into a pointcloud using camera intrinsic and depthmap ([3,x*y])
    deprojected = (np.linalg.inv(K_sc) @ sc_mesh_homo) * depth_flattened

    ##### Step 2-4
    # make the pointcloud coordinate homogeneous 3D coordinate ([4,x*y])
    pcd_homo = np.concatenate((deprojected,np.ones((1,x_sc*y_sc))),axis=0)

    ##### Step 2-5
    # transform the pointcloud into the robot base frame ([4,x*y])
    # note: i transform to rgb frame for rendering
    points_world = T_relative @ pcd_homo

    ##### Step 2-6
    # transpose the pointclouds and discard the homogenous part ([x*y,3])
    points_world = np.delete(points_world,-1,0)
    points_world = np.transpose(points_world)
    return points_world

def get_depth_error(gt,measured):
    error = np.abs(gt-measured)
    # Set error of invalid areas to zero 
    out = np.where(measured == 0.0, np.zeros_like(measured),error)
    # use this one if the error in the invalid areas are not to be calculated:
    # return np.absolute(np.ma.masked_where(measured == 0.0, gt) - np.ma.masked_where(measured == 0.0, measured))
    return out


def main():

    ############################################################################
    #####        loading necessary files. do not modify anything           #####
    ############################################################################

    # ee_pose
    ee_to_rb = np.loadtxt("ee_to_rb.txt")

    # loading d435 info
    depth_d435 = cv2.imread(os.path.join("d435", "000000.png"), -1) / 1000
    intrinsic_d435 = np.loadtxt(os.path.join("d435", "intrinsics.txt"))
    d435_to_ee = ros_to_mat(np.loadtxt(os.path.join("d435","d435_to_ee.txt")))

    # loading l515 info
    depth_l515 = cv2.imread(os.path.join("l515", "000000.png"), -1) / 1000
    intrinsic_l515 = np.loadtxt(os.path.join("l515", "intrinsics.txt"))
    l515_rgb_to_ee = ros_to_mat(np.loadtxt(os.path.join("l515","l515_rgb_to_ee.txt")))
    l515_depth_to_l515_rgb = np.loadtxt(os.path.join("l515","l515_depth_to_l515_rgb.txt"))

    # loading rgb info # Note: this is actually a single channel depth map from the point of view of the rgb camera. NOT an rgb image.
    rgb = cv2.imread(os.path.join("rgb", "000000.png"), -1) / 1000
    rgb_to_ee = ros_to_mat(np.loadtxt(os.path.join("rgb","pol_to_ee.txt")))
    intrinsic_rgb = np.loadtxt(os.path.join("rgb","intrinsics.txt"))

    ############################################################################
    #####                  Your part starts from here !                    #####
    ############################################################################

    camera = "d435"

    # create pyrender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0])

    ####### Task 1. #######
    # Obtain both RGB and Depth camera’s pose to robot base

    ##### Step 1-1
    # obtain rgb camera pose to robot base (rgb_to_rb)
    rgb_to_rb = ee_to_rb @ rgb_to_ee

    if camera == "d435":
        # try to use d435 first
        depth = depth_d435
        depth_intrinsic = intrinsic_d435

        ##### Step 1-2
        # obtain depth camera pose to robot base (depth_to_rb) with d435
        depth_to_rb = ee_to_rb @ d435_to_ee


    elif camera == "l515":

        ####### Task 5. #######
        # after finishing the task with d435, try with l515
        depth = depth_l515
        depth_intrinsic = intrinsic_l515

        ##### Step 5-1
        # obtain depth camera pose to robot bose (depth_to_rb) with l515
        # note that l515's hand-eye calibration is obtained from RGB.
        # use l515 rgb <-> depth calibration matrix to obtain depth_to_rb
        depth_to_rb = ee_to_rb @ l515_rgb_to_ee @ l515_depth_to_l515_rgb

    ##### Step 1-3
    # obtain relative camera pose from depth to rgb (depth_to_rb)
    depth_to_rgb = invert_transform(rgb_to_rb) @ depth_to_rb

    ####### Task 2. #######
    # Implement deprojection function which converts the depthmap into a pointcloud and transform it into the RGB
    # camera’s viewpoint
    # in: depth image of selected depth camera
    # in: transformation from depth to world frame,
    # in: intrinsics matrix of the depth camera
    points_world = deproj(depth,depth_to_rgb,depth_intrinsic)

    ##### Step 2-7
    # now pointcloud is centered on the rgb camera's viewpoint
    # convert pointcloud's orientation into pyrender camera's orientation by flipping axis
    # note that if pointcloud is not centered in rgb camera (e.g. centered on robot base), more steps are required and
    # we dont consider that case in this challenge.

    points_world[:,1:] *= -1

    ####### Task 3. #######
    # Convert the points into spheres with volume

    ##### Step 3-1
    # convert each point into a sphere with radius = 2mm (0.002m)
    # each point will have identity as rotation and its x,y,z coordinate as translation
    # check here for generating spheres from the points :
    # https://pyrender.readthedocs.io/en/latest/examples/models.html
    sm = trimesh.creation.uv_sphere(radius=0.002)
    sm.visual.vertex_colors = [1.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (points_world.shape[0], 1, 1))
    tfs[:,:3,3] = points_world
    m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

    ##### Step 3-2
    # add the sphere's mesh. If the all instructions are followed properly, the mesh center will be located
    # at the origin
    scene.add(m)

    ####### Task 4. #######
    # Render the depthmap with the spheres and compare the quality of the warped depthmap with the GT from the
    # assignment 2

    ##### Step 4-1
    # setup pyrender camera with rgb info
    fx = intrinsic_rgb[0,0]
    fy = intrinsic_rgb[1,1]
    cx = intrinsic_rgb[0,2]
    cy = intrinsic_rgb[1,2]
    camera = pyrender.IntrinsicsCamera(fx,fy,cx,cy)
    camera_node = pyrender.Node(camera=camera)

    ##### Step 4-2
    # Add the camera node
    scene.add_node(camera_node)

    ##### Step 4-3
    # render the scene
    scene.ambient_light = [1.,1.,1.]
    height, width = rgb.shape
    r = pyrender.OffscreenRenderer(width,height)
    _, depth_warped = r.render(scene)

    ##### Step 4-4
    # load rendered GT from the assignment 2 and examine the absolute error of the depth modality
    # use 1x3 plot with depth_warped (1,1), depth_gt (1,2), depth_error (1,3)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(depth_warped)
    plt.subplot(1, 3, 2)
    plt.imshow(rgb)
    plt.subplot(1, 3, 3)
    plt.imshow(get_depth_error(rgb,depth_warped))
    plt.show()


if __name__ == "__main__":
    main()