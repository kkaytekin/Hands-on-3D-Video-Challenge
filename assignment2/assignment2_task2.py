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

def main():

    ############################################################################
    #####        loading necessary files. do not modify anything           #####
    ############################################################################

    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1., 1., 1.])

    ## loading meshes
    obj_name = "teapot_animal"
    fname = os.path.join("./obj_meshes", obj_name, obj_name+".obj")
    trimesh_obj = trimesh.load(fname)
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    mesh_node = scene.add(mesh)

    ## setting up the camera for pyrender
    rgb = plt.imread("000000.png")
    h,w,c = rgb.shape

    intrinsic = np.loadtxt("intrinsic.txt")
    fx = intrinsic[0,0]
    fy = intrinsic[1,1]
    cx = intrinsic[0,2]
    cy = intrinsic[1,2]
    camera = pyrender.IntrinsicsCamera(fx,fy,cx,cy)
    camera_node = pyrender.Node(camera=camera)
    scene.add_node(camera_node)
    r = pyrender.OffscreenRenderer(w, h)

    ############################################################################
    #####                  Your part starts from here !                    #####
    ############################################################################

    ####### Task 2. #######
    # Find out the orientation of Pyrender by rendering the object multiple times.

    ##### Step 1
    # Check out the plots, rendered object with with z = 1m VS -1m
    # Which direction does Pyrender point for z axis?

    # render positive z
    mesh_pose = np.identity(4)
    mesh_pose[:3, 3] = [0, 0, 1]
    scene.set_pose(mesh_node, mesh_pose)
    color_pos_z, depth = r.render(scene)

    # render negative z
    mesh_pose = np.identity(4)
    mesh_pose[:3, 3] = [0, 0, -1]
    scene.set_pose(mesh_node, mesh_pose)
    color_neg_z, depth = r.render(scene)

    # compare the plot. Which direction is z?
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(color_pos_z)
    # plt.subplot(1, 2, 2)
    # plt.imshow(color_neg_z)
    # plt.show()

    # Answer : The negative direction.

    ##### Step 2
    # Then place the object in front of the camera (from step 1) either (0,0,1) or (0,0,-1) depending on the visibility
    # and test with x,y axis in the similar way to figure out which direction does Pyrender point for x,y axis
    # Here, make sure to set x,y with relatively smaller number (i.e. 0.2) So that the rendering doesn't go out of
    # the image boundary

    # render positive x
    mesh_pose = np.identity(4)
    mesh_pose[:3, 3] = [0.5, 0, -1]
    scene.set_pose(mesh_node, mesh_pose)
    color_pos_x, depth = r.render(scene)

    # render negative x
    mesh_pose = np.identity(4)
    mesh_pose[:3, 3] = [-0.05, 0, -1]
    scene.set_pose(mesh_node, mesh_pose)
    color_neg_x, depth = r.render(scene)

    # compare the plot. Which direction is x?
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(color_pos_x)
    # plt.subplot(1, 2, 2)
    # plt.imshow(color_neg_x)
    # plt.show()
    
    # render positive y
    mesh_pose = np.identity(4)
    mesh_pose[:3, 3] = [0.5, 0.5, -1]
    scene.set_pose(mesh_node, mesh_pose)
    color_pos_y, depth = r.render(scene)

    # render negative y
    mesh_pose = np.identity(4)
    mesh_pose[:3, 3] = [0.5, -0.5, -1]
    scene.set_pose(mesh_node, mesh_pose)
    color_neg_y, depth = r.render(scene)
    # compare the plot. Which direction is y?
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(color_pos_y)
    plt.subplot(1, 2, 2)
    plt.imshow(color_neg_y)
    plt.show()    

    ####  Now you know the orientation of Pyrender :)
    # x stays the same, z and y are reversed. 
    # which means Pyrender orientation is 180 degrees rotated version (about x axis) of the usual camera frame we have.
    # x positive: slide right ,y negative: slide downwards ,z negative: get further away.

if __name__ == "__main__":
    main()






