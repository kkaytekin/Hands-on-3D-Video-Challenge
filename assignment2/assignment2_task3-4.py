import pyrender, trimesh, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from assignment2_task1 import invert_transform
import imageio

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

    scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[0.3, 0.3, 0.3])

    ## loading meshes
    object_name_list = os.listdir("./obj_meshes")
    print("loading objects in the scene..")
    mesh_nodes = []

    for each_obj_name in object_name_list:
        fname = os.path.join("./obj_meshes",each_obj_name,each_obj_name+".obj")
        pose = np.loadtxt(os.path.join("./obj_meshes",each_obj_name,each_obj_name+".txt"))

        trimesh_obj = trimesh.load(fname)
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        mesh_node = scene.add(mesh)

        mesh_nodes.append({"obj_name": each_obj_name,
                           "node": mesh_node,
                           "obj_to_RB":pose})
        print(each_obj_name + " loaded")

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

    ####### Task 3. #######
    # Find out the conversion factor and then render the entire scene as a depthmap

    rgb_to_EE = ros_pose_to_mat_pose(np.loadtxt("pol_to_ee.txt"))
    EE_to_RB = np.loadtxt("000000.txt")

    for each_mesh_node in mesh_nodes:

        ##### Step 3-1
        # Lets first obtain relative pose between the camera and each object "obj_to_rgb" like in Step 1-1.

        obj_to_RB = each_mesh_node["obj_to_RB"]
        obj_to_rgb = invert_transform(rgb_to_EE) @ invert_transform(EE_to_RB) @ obj_to_RB

        ##### Step 3-2
        # Calculate the conversion factor from task1 and task2's orientations and apply on the "obj_to_rgb"
        factor = np.eye(4)
        factor[1,1] = -1
        factor[2,2] = -1

        obj_to_rgb = np.matmul(factor, obj_to_rgb)

        node = each_mesh_node["node"]
        scene.set_pose(node, obj_to_rgb)

    color,depth = r.render(scene)

    ##### Step 3-3
    # Save the depthmap as 16 bit greyscale image (scale 1m distance as 1000)
    depth = 1000*depth
    print("Max value: ",np.max(depth))
    print("Min value: ",np.min(depth))
    imageio.imwrite("my_depth_image.png",depth.astype('uint16'))
    # plt.figure()
    # plt.imshow(depth)
    # plt.show()

    ####### Task 4. #######
    # Obtain the mask of the object to verify the quality of the object annotation

    ##### Step 4-1
    # first remove the object mesh node of background and table
    # (you can check out the documentation https://pyrender.readthedocs.io/en/latest/generated/pyrender.Scene.html)
    for each_mesh_node in mesh_nodes:
        if "background" in each_mesh_node["obj_name"]:
            scene.remove_node(each_mesh_node["node"])
    # Note: instead of for loop, I could do this operation for mesh_nodes[0] and mesh_nodes[1]-
    # The loop is slower but easier to understand.

    ##### Step 4-2
    # render the scene without the background and table
    _,depth_wo_bg = r.render(scene)
    # plt.figure()
    # plt.imshow(depth_wo_bg)
    # plt.show()

    ##### Step 4-3
    # create the mask from the rendering
    # hint : only background and table will have value (0,0,0) in both RGB and depth rendering
    mask = np.ma.masked_where(depth_wo_bg == 0.0, depth_wo_bg)

    ##### Step 4-5
    # augment the object mask on the rgb image with any noticeable color in the semi-transparent way

    # mask[mask>0] = 1 # we can set to single colour if we want...

    ##### Step 4
    # plot RGB, depth, augmented RGB in the same figure as a subplot

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(depth)
    plt.subplot(1,3,2)
    plt.imshow(rgb)
    plt.subplot(1,3,3)
    plt.imshow(rgb)
    plt.imshow(mask, cmap='jet', interpolation='none', alpha = 0.7)
    plt.show()

if __name__ == "__main__":
    main()






