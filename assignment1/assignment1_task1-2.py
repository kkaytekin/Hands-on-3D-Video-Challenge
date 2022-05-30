from operator import invert
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

def proj_points(points,intrinsic,camera_pose):
    n = points.shape[0]
    ##### Step 2-3
    # make the points into homogeneous coordinate (n,4)
    points = np.concatenate((points,np.ones((n,1))),axis=1)

    ##### Step 2-4
    # transpose the points to make the pose multiplication easier (4,n)
    points = points.transpose()

    ##### Step 2-5
    # apply camera pose (4,n)
    C_points = camera_pose @ points

    ##### Step 2-6
    # project onto the camera image plane and remove the homogenous points (3,n)
    proj_points = np.ones((3,n))
    proj_points[0,:] = C_points[0,:] / C_points[2,:]
    proj_points[1,:] = C_points[1,:] / C_points[2,:]

    ##### Step 2-7
    # convert the coordinate of image plane into image pixel coordinate by using intrinsic k
    pxl_points = intrinsic @ proj_points

    ##### Step 2-8
    # remove the homogeneous coordinate and transpose back to (n,2) format
    points_cam_pixel = pxl_points.transpose()

    return points_cam_pixel

def invert_transform(transform):
    out = np.identity(4)
    rot_transpose = transform[:3,:3].transpose()
    trans = transform[:3,3]
    out[:3,:3] = rot_transpose
    out[:3, 3] = -np.matmul(rot_transpose,trans)
    return out

def main():

    ####### Task 1. #######
    # Form the relative pose graph from Fig 1. and calculate the Hand-eye-calibration matrix (cam_to_ee)
    marker_to_rb = ros_pose_to_mat_pose(np.loadtxt("checkerboard_to_rb.txt"))
    marker_to_cam = np.loadtxt("checkerboard_to_cam.txt")
    ee_to_rb = np.loadtxt("ee_to_rb.txt")

    ##### Step 1
    # Check the Fig.1 for the direction. Form a relative pose graph if necessary. Note that traversing the graph in the
    # inverse direction is equivalent to using inverse of the given pose matrix.
    cam_to_marker = invert_transform(marker_to_cam)
    rb_to_ee = invert_transform(ee_to_rb)
    cam_to_ee = rb_to_ee @ marker_to_rb @ cam_to_marker

    ####### Task 2. #######
    # Project the tooltip’s 3D locations on to given image by using given camera intrinsic
    points = np.loadtxt("point_measured.txt")
    intrinsic = np.loadtxt("intrinsic.txt")

    ##### Step 2-1
    # find out the transformation matrix between the measured points and the camera
    rb_to_marker = invert_transform(marker_to_rb)
    rb_to_cam = marker_to_cam @ rb_to_marker


    ##### Step 2-2
    # implement point projection function
    pixel_coords = proj_points(points,intrinsic,rb_to_cam)

    img = plt.imread("000000.png")

    ##### Step 2-9
    # plot the figure and overlay the projected points
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(pixel_coords[:,0],pixel_coords[:,1],c='r')
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    main()