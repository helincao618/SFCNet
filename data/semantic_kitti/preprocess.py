"""
Preprocess the Semantic Kitti dataset to generate the following files:
    -depthmask: depth mask for each frame
    -semanticmask: semantic mask for each frame
    -sparsedepthmask: sparse depth mask for each frame
    -depthimage: depth image for each frame
    -semanticimage: semantic image for each frame
    -label: label for each frame with scale 1/1, 1/2, 1/4, 1/8
    -invalid: invalid mask for each frame with scale 1/1, 1/2, 1/4, 1/8

This code is based on the original Semantic Kitti dataset preprocessing code

Author: Helin Cao
Date: YYYY-MM-DD
License: MIT License (or any other license you want to use)
"""
import glob
import math
import os
import sys
import yaml
import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from dvis import dvis

repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)
import semantic_kitti.io_data as SemanticKittiIO

# Constants
VISUALIZE = False
CAM2TOCAM0 = np.array([[1,0,0,-0.06],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
R0_RECT = np.array([[9.999239000000e-01,9.837760000000e-03,-7.445048000000e-03,0],
                    [-9.869795000000e-03,9.999421000000e-01,-4.278459000000e-03,0],
                    [7.402527000000e-03,4.351614000000e-03,9.999631000000e-01,0],
                    [0,0,0,1]])
DOWNSCALE = [1, 8]
# Functions definition
def _downsample_label(label, invalid, voxel_size, downscale):
    if downscale == 1:
        return label, invalid
    ds = downscale
    small_size = (
        voxel_size[0] // ds,
        voxel_size[1] // ds,
        voxel_size[2] // ds,
    )  # small size
    label_downscale = np.zeros(small_size, dtype=np.uint8)
    empty_t = 0.95 * ds * ds * ds  # threshold
    s01 = small_size[0] * small_size[1]
    label_i = np.zeros((ds, ds, ds), dtype=np.int32)

    for i in range(small_size[0] * small_size[1] * small_size[2]):
        z = int(i / s01)
        y = int((i - z * s01) / small_size[0])
        x = int(i - z * s01 - y * small_size[0])

        label_i[:, :, :] = label[
            x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
        ]
        label_bin = label_i.flatten()

        zero_count_0 = np.array(np.where(label_bin == 0)).size
        zero_count_255 = np.array(np.where(label_bin == 255)).size

        zero_count = zero_count_0 + zero_count_255
        if zero_count > empty_t:
            label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
        else:
            label_i_s = label_bin[
                np.where(np.logical_and(label_bin > 0, label_bin < 255))
            ]
            label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
    invalid = np.zeros_like(label)
    invalid[np.isclose(label, 255)] = 1
    return label_downscale, invalid

def majority_pooling(grid, k_size=2):
    result = np.zeros(
        (grid.shape[0] // k_size, grid.shape[1] // k_size, grid.shape[2] // k_size)
    )
    for xx in range(0, int(np.floor(grid.shape[0] / k_size))):
        for yy in range(0, int(np.floor(grid.shape[1] / k_size))):
            for zz in range(0, int(np.floor(grid.shape[2] / k_size))):
                sub_m = grid[
                    (xx * k_size) : (xx * k_size) + k_size,
                    (yy * k_size) : (yy * k_size) + k_size,
                    (zz * k_size) : (zz * k_size) + k_size,
                ]
                unique, counts = np.unique(sub_m, return_counts=True)
                if True in ((unique != 0) & (unique != 255)):
                    # Remove counts with 0 and 255
                    counts = counts[((unique != 0) & (unique != 255))]
                    unique = unique[((unique != 0) & (unique != 255))]
                else:
                    if True in (unique == 0):
                        counts = counts[(unique != 255)]
                        unique = unique[(unique != 255)]
                value = unique[np.argmax(counts)]
                result[xx, yy, zz] = value
    return result

def dot(transform, pts):
    if pts.shape[1] == 3:
        pts = np.concatenate([pts,np.ones((len(pts),1))],1)
    return (transform @ pts.T).T

def project_size(depth):
    return math.ceil(3000/(depth+15)**2)

def img2point(u, v, d, P):
    # Create a homogeneous image coordinate
    uv1 = np.array([u, v, 1])
    # Compute the homogeneous 3D point in camera coordinates
    X_c = np.linalg.pinv(P) @ (d * uv1)
    # Normalize the homogeneous coordinate by dividing by the last element
    X_c /= X_c[-1]
    return X_c[:-1]

def filter_outside_points(valid_labels, pos_img, imgw, imgh):
    # chop off the points that are outside the image
    valid_labels = valid_labels[pos_img[:, 0] > 0]
    pos_img = pos_img[pos_img[:, 0] > 0]

    valid_labels = valid_labels[pos_img[:, 0] < imgw]
    pos_img = pos_img[pos_img[:, 0] < imgw]

    valid_labels = valid_labels[pos_img[:, 1] > 0]
    pos_img = pos_img[pos_img[:, 1] > 0]

    valid_labels = valid_labels[pos_img[:, 1] < imgh]
    pos_img = pos_img[pos_img[:, 1] < imgh]

    return valid_labels, pos_img

# Main function
@hydra.main(config_name='../../config/sfc.yaml')
def main(config: DictConfig):
    splits = {
            'train': ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10'],
            'val': ['08'],
            'test': ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
        }
    scene_size =[256, 256, 32]
    imgh, imgw = 376, 1241
    split = 'val'
    sequences = splits[split]
    config_path = os.path.join(
            get_original_cwd(),
            'SFCNet',
            'SFCNet',
            'config',
            'semantic-kitti.yaml',
        )
    remap_lut = SemanticKittiIO.get_remap_lut(config_path)
    learning_map_inv = SemanticKittiIO.get_inv_map(config_path)
    sem_pallete = yaml.safe_load(open(config_path, 'r'))["color_map"]

    # Loop through sequences
    for sequence in sequences:
        
        # Read paths and store in lists
        voxel_path = os.path.join(
            config.kitti_voxel_root, 'dataset', 'sequences', sequence
        )
        velodyne_path = os.path.join(
            config.kitti_pointcloud_root, 'dataset', 'sequences', sequence,
        )
        if split == 'train' or split == 'val':
            label_paths = sorted(
                glob.glob(os.path.join(voxel_path, 'voxels', '*.label'))
            )
            invalid_paths = sorted(
                glob.glob(os.path.join(voxel_path, 'voxels', '*.invalid'))
            )
            ptslabel_paths = sorted(
                glob.glob(os.path.join(velodyne_path, 'labels', '*.label'))
            )
            # Create output directories
            label_outdir = os.path.join(config.kitti_preprocess_root, sequence, 'labels')
            invalid_outdir = os.path.join(config.kitti_preprocess_root, sequence, 'invalid')
            semanticimg_outdir = os.path.join(config.kitti_preprocess_root, sequence, 'semanticimg')
            semanticmask_outdir = os.path.join(config.kitti_preprocess_root, sequence, 'semanticmask')
            os.makedirs(label_outdir, exist_ok=True)
            os.makedirs(invalid_outdir, exist_ok=True)
            os.makedirs(semanticimg_outdir, exist_ok=True)
            os.makedirs(semanticmask_outdir, exist_ok=True)

        pts_paths = sorted(
            glob.glob(os.path.join(velodyne_path, 'velodyne', '*.bin'))
        )
        poses = SemanticKittiIO.read_poses_SemKITTI(os.path.join(voxel_path, 'poses.txt'))
        calib = SemanticKittiIO.read_calib_SemKITTI(os.path.join(voxel_path, 'calib.txt'))
        p2 = calib['P2']
        T_velo_2_cam = calib['Tr']

        # Create output directories
        sparsedepthimg_outdir = os.path.join(config.kitti_preprocess_root, sequence, 'sparsedepthimg')
        sparsedepthmask_outdir = os.path.join(config.kitti_preprocess_root, sequence, 'sparsedepthmask')
        os.makedirs(sparsedepthimg_outdir, exist_ok=True)
        os.makedirs(sparsedepthmask_outdir, exist_ok=True)

        for i in tqdm(range((len(pts_paths)-1)//5+1)):
            frame_id, extension = os.path.splitext(os.path.basename(pts_paths[5*i]))
            ## Semantic and depth mask and image
            # initialize image and mask
            depthmask = np.ones([imgh,imgw], dtype = int)*200 #default
            sparsedepthmask = np.ones([imgh,imgw], dtype = int)*200
            # point cloud from velodyne
            pose_velo = np.linalg.inv(T_velo_2_cam).dot(poses[5*i].dot(T_velo_2_cam))
            scan = SemanticKittiIO.read_pointcloud_SemKITTI(pts_paths[5*i])
            scan_global = dot(pose_velo, scan)
            pose_cam2 = np.linalg.inv(CAM2TOCAM0).dot(np.linalg.inv(poses[5*i]).dot(T_velo_2_cam))
            
            if split == 'train' or split == 'val':
                
                ## Reading and Downscaling labels and invalid masks
            
                LABEL = SemanticKittiIO.read_label_SemKITTI(label_paths[i])
                INVALID = SemanticKittiIO.read_invalid_SemKITTI(invalid_paths[i])
                LABEL = remap_lut[LABEL.astype(np.uint16)].astype(np.float32)  # Remap 20 classes semanticKITTI SSC
                LABEL[np.isclose(INVALID, 1)] = 255  # Setting to unknown all voxels marked on invalid mask...
                LABEL = LABEL.reshape(scene_size)
                for scale in DOWNSCALE:
                    filename = frame_id + '_' + str(scale) + '.npy'
                    label_filename = os.path.join(label_outdir, filename)
                    invalid_filename = os.path.join(invalid_outdir, filename)
                    # If files have not been created...
                    if not os.path.exists(label_filename) & os.path.exists(invalid_filename):
                        LABEL_ds, INVALID_ds = _downsample_label(LABEL, INVALID, scene_size, scale)
                        np.save(label_filename, LABEL_ds)
                        np.save(invalid_filename, INVALID_ds)
                        # print('wrote to', label_filename)
                        # print('wrote to', invalid_filename)

                # point cloud from GT
                semanticimg = np.zeros([imgh,imgw,3], dtype = int)
                semanticmask = np.zeros([imgh,imgw], dtype = int)
                # point cloud semantic color
                semcolor = np.zeros([len(scan),3])
                # label
                labels_velo = SemanticKittiIO.read_pc_label_SemKITTI(ptslabel_paths[5*i])
                for idx, label in enumerate(labels_velo):
                    labels_velo[idx] = label & 0xFF #take out lower 16 bits                  
                    if labels_velo[idx] in sem_pallete:
                        semcolor[idx] = np.asarray(sem_pallete[labels_velo[idx]][::-1]) #reverse the list from bgr to rgb
                    else:
                        semcolor[idx] = np.array([0,0,0])
                labels_gt = np.fromfile(label_paths[i], dtype=np.uint16).reshape(scene_size)
                valid_label_mask = (labels_gt>39) & (labels_gt <100) # only keep static object
                valid_labels = labels_gt[valid_label_mask]
                valid_label_inds = np.stack(np.nonzero(valid_label_mask),1)
                vox2pts = np.eye(4)
                vox2pts[:3,:3] = np.diag([1/5,1/5,1/5])
                vox2pts[:3,3] = np.array([0.0,-25.6,-2.0])          
                pts_gt = dot(vox2pts, valid_label_inds)
                pts_gt_global = pts_gt.dot(pose_velo.T)[:,0:3]
                # voxel color
                valid_colors = np.zeros((len(pts_gt),3))
                for label in np.unique(valid_labels):
                    if label>0:
                        label_mask = valid_labels == label
                        valid_colors[label_mask] = sem_pallete[label][::-1]
                scan_gt_global = np.concatenate([scan_global[:,:3],pts_gt_global],0)
                valid_labels = np.concatenate([labels_velo,valid_labels],0)
                scan_gt_semcolor = np.concatenate([semcolor, valid_colors],0)
                scan_gt_semcolor_global = np.concatenate([scan_gt_global, scan_gt_semcolor],1)
                if VISUALIZE:
                    dvis(scan_gt_semcolor_global, l=3, t=i, vs=1/5, name='volume/semantic volume'+ str(i))
                    dvis(np.concatenate([scan_global[:,:3], semcolor],1), l=1, t=i, vs=1/5, name='volume/semantic volume'+ str(i))
                    dvis(np.concatenate([pts_gt_global, valid_colors],1), l=2, t=i, vs=1/5, name='volume/semantic volume'+ str(i))
                
                # project to image
                
                pos_scans_cam = dot(pose_cam2, scan_gt_global)
                valid_labels = valid_labels[pos_scans_cam[:,2]>0]
                pos_scans_cam = pos_scans_cam[pos_scans_cam[:,2]>0]
                pos_img = p2 @ R0_RECT @ pos_scans_cam.T
                pos_img[:2] /= pos_img[2,:]
                pos_img = pos_img.T
                valid_labels, pos_img = filter_outside_points(valid_labels, pos_img, imgw, imgh)
                # sparse to dense
                for idx, posimg in enumerate(pos_img):
                    if (posimg[2] < depthmask[int(posimg[1]), int(posimg[0])] and
                        valid_labels[idx] in sem_pallete.keys() and
                        int(posimg[1] - project_size(posimg[2])) >= 0 and
                        int(posimg[1] + project_size(posimg[2])) <= imgh and
                        int(posimg[0] - project_size(posimg[2])) >= 0 and
                        int(posimg[0] + project_size(posimg[2])) <= imgw):
                        semanticmask[int(posimg[1] - project_size(posimg[2])):int(posimg[1] + project_size(posimg[2])),
                                        int(posimg[0] - project_size(posimg[2])):int(posimg[0] + project_size(posimg[2]))] = valid_labels[idx]
                        # depthmask[int(posimg[1] - project_size(posimg[2])):int(posimg[1] + project_size(posimg[2])),
                        #             int(posimg[0] - project_size(posimg[2])):int(posimg[0] + project_size(posimg[2]))] = posimg[2]
                # map the mask to learning mask
                semanticmask = remap_lut[semanticmask]  # Remap 20 classes semanticKITTI SSC
                # save the mask array
                np.save(semanticmask_outdir +'/'+ frame_id +'.npy',semanticmask.astype(np.uint8))
                # write the img
                # for label in np.unique(semanticmask):
                #     if label>0 and label<255:
                #         label_mask = semanticmask == label
                #         semanticimg[label_mask] = sem_pallete[learning_map_inv[label]][::-1] #reverse the list from bgr to rgb
                # img = Image.fromarray(semanticimg.astype(np.uint8)).convert('RGB')
                # img.save(semanticimg_outdir +'/'+ frame_id+'.png')
            
            
            # generate sparse depth mask fuse 5 frame
            fused_scan_global = np.array([[0,0,0,0]])
            for j in range(1):
                if 5*i+j <len(pts_paths):
                    single_scan = SemanticKittiIO.read_pointcloud_SemKITTI(pts_paths[5*i+j])
                    pose_single_velo = np.linalg.inv(T_velo_2_cam).dot(poses[5*i+j].dot(T_velo_2_cam))
                    scan_global = dot(pose_single_velo, single_scan)
                    fused_scan_global = np.concatenate([fused_scan_global,scan_global],0)
            pos_scans = dot(pose_cam2, fused_scan_global[:,:3])
            pos_scans = pos_scans[pos_scans[:,2]>0]
            pos_img_scans = p2 @ R0_RECT @ pos_scans.T
            pos_img_scans[:2] /= pos_img_scans[2,:]
            pos_img_scans = pos_img_scans.T
            pos_img_scans = pos_img_scans[pos_img_scans[:, 0] > 0]
            pos_img_scans = pos_img_scans[pos_img_scans[:, 0] < imgw]
            pos_img_scans = pos_img_scans[pos_img_scans[:, 1] > 0]
            pos_img_scans = pos_img_scans[pos_img_scans[:, 1] < imgh]
            for idx, pos_img_scan in enumerate(pos_img_scans):
                # sparsedepthmask[(int(pos_img_scan[1])-1):(int(pos_img_scan[1])+1),(int(pos_img_scan[0])-1):(int(pos_img_scan[0])+1)] = pos_img_scan[2]
                sparsedepthmask[int(pos_img_scan[1]),int(pos_img_scan[0])] = pos_img_scan[2]
            # save the mask array
            # np.save(sparsedepthmask_outdir +'/'+ frame_id +'.npy',sparsedepthmask.astype(np.float32))
            # draw on img
            cmap = plt.cm.jet
            norm_depth_map = (sparsedepthmask - 0) / (100 - 0)
            norm_depth_map[sparsedepthmask == 200] = 0
            color_image = cmap(norm_depth_map)
            color_image[sparsedepthmask == 200, 0] = 1.0
            color_image[sparsedepthmask == 200, 1] = 1.0
            color_image[sparsedepthmask == 200, 2] = 1.0
            plt.imsave(sparsedepthimg_outdir +'/'+ frame_id+'.png', color_image)

if __name__ == '__main__':
    main()