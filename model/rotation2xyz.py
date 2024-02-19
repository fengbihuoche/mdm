# This code is based on https://github.com/Mathux/ACTOR.git
import torch
import utils.rotation_conversions as geometry
import numpy as np
from os.path import join as pjoin
import numpy as np
import os

from model.smpl import SMPL, JOINTSTYPE_ROOT
# from .get_model import JOINTSTYPES
JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]
from smplx import SMPLLayer
import pickle
from data_loaders.humanml.common.skeleton import Skeleton
from data_loaders.humanml.common.quaternion import *
from data_loaders.humanml.utils.paramUtil import *


example_id = "010225"
# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22
data_dir = '/remote-home/share/joints/'
example_data = np.load("/storage/group/4dvlab/congpsh/Diff-Motion/010225.npy")
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
# print(tgt_skel.shape)
# (joints_num, 3)
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])  

def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    
    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt
    
    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)
    
    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    # print(new_joints[0]-positions[0])
    # print(new_joints[0].shape)
    return new_joints

    
class Rotation2xyz:
    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL().eval().to(device)
        
    def __call__(self, x, mask, pose_rep, translation, glob,
                 jointstype, vertstrans, betas=None, beta=0,
                 glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == "xyz":
            return x

        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)

        if not glob and glob_rot is None:
            raise TypeError("You must specify global rotation if glob is False")

        if jointstype not in JOINTSTYPES:
            raise NotImplementedError("This jointstype is not implemented.")
        
        print(x.shape)

        if translation:
            x_translations = x[:, -1, :3] #1x3xframes
            x_rotations = x[:, :-1]
        else:
            x_rotations = x
            
        

        print(x_translations.shape)
        print(x_rotations.shape)
          
        x_rotations = x_rotations.permute(0, 3, 1, 2)

        nsamples, time, njoints, feats = x_rotations.shape

        # Compute rotations (convert only masked sequences output)
        if pose_rep == "rotvec":
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == "rotmat":
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == "rotquat":
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == "rot6d":
            rotations = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(x_rotations[mask]))
            print(rotations.shape) ####
        else:
            raise NotImplementedError("No geometry for this one.")

        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0].reshape(40,3)
            rotations = rotations[:, 1:].reshape(40,69)
            print(global_orient.shape)
            print(rotations.shape)

        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas],
                                dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
            # import ipdb; ipdb.set_trace()

        #self.smpl_model=self.smpl_model.double()
        
       
       #print(global_orient.shape)
        #rotations=torch.matmul(rotations, r.unsqueeze(0).unsqueeze(0).expand(rotations.size()))
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)

        # get the desirable joints
        joints = out[jointstype] #jointstype="smpl" or "vertices" : joint nframesx24x3 / vertices nframes x 6890 x 3
        
        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints #in fact vertex
        
              

        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous() #nsamples x 6890 x 3 x 40
        
        originmotions=pickle.load(open(pjoin("/storage/group/4dvlab/congpsh/Diff-Motion/","train_split_motion.pkl"),"rb"))  ####originmotions
        index=2292
        joint=originmotions[index]["joints"]
        theta = -np.pi / 2
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        joint = joint - joint[0,0]
        joint=joint[:,:22] @ rotation_matrix.T 
        positions =uniform_skeleton(joint, tgt_offsets)
        floor_height = positions.min(axis=0).min(axis=0)[1]
        print(floor_height)

        # the first translation root at the origin on the prediction
        if jointstype != "vertices":
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]

        if translation and vertstrans:
            # the first translation root at the origin
            x_translations = x_translations - x_translations[:, :, [0]]  #000

            # add the translation to all the joints
            x_xyz = x_xyz + x_translations[:, None, :, :]
            #print(x_xyz.shape)
            for i in range(x_xyz.shape[0]):
                x_xyz[i,:,1,:]-=floor_height

        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz
