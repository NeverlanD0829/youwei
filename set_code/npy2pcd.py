import numpy as np
import open3d as o3d

def depth_to_point_cloud(depth_array, fx, fy, cx, cy):
    height, width = depth_array.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_array
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def npy_to_pcd(input_file, output_file, fx, fy, cx, cy):
    # Load numpy array from .npy file
    depth_array = np.load(input_file).astype(np.float32)
    
    # Convert depth image to point cloud
    points = depth_to_point_cloud(depth_array, fx, fy, cx, cy)
    
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Save point cloud to .pcd file
    o3d.io.write_point_cloud(output_file, pcd)
    print(f"Point cloud saved as {output_file}")

if __name__ == "__main__":
    input_npy_file = "/media/chen/Chen Disk/Weight_Dataset/Dataset1/1/depth/0.npy"
    output_pcd_file = "/home/chen/Desktop/data/Dataset_Point_cloud/point_cloud.pcd"
    
    # Camera intrinsic parameters (example values, need to be adjusted according to your camera)
    fx = 525.0  # Focal length in x axis
    fy = 525.0  # Focal length in y axis
    cx = 319.5  # Principal point in x axis
    cy = 239.5  # Principal point in y axis

    npy_to_pcd(input_npy_file, output_pcd_file, fx, fy, cx, cy)
