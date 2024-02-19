import os
import torch
import torch.nn as nn
import json
import numpy as np
import copy
import glob
from collections import defaultdict
from omegaconf import DictConfig
from sklearn.metrics import pairwise_distances
from chamfer_distance import ChamferDistance as chamfer_dist
from os.path import join as pjoin
import torch.nn.functional as F
import pickle
import sys
sys.path.append("/public/home/wangzy17/motion-diffusion-model/utils")
sys.path.append(r"/public/home/wangzy17/Scene-Diffuser/utils/")
sys.path.append(r"/public/home/wangzy17/motion-diffusion-model/data_loaders/humanml/common")
sys.path.append(r"/public/home/wangzy17/motion-diffusion-model/data_loaders/humanml/utils")
from smplx_utils import SMPLXWrapper
#from utils.smplx_utils import get_marker_indices, smplx_signed_distance
from skeleton import Skeleton
from quaternion import *
from paramUtil import *


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

class SMPLXGeometry():

    def __init__(self, body_segments_dir: str) -> None:
        ## load contact part
        self.contact_verts_ids = {}
        self.contact_faces_ids = {}

        part_files = glob.glob(os.path.join(body_segments_dir, '*.json'))
        for pf in part_files:
            with open(pf, 'r') as f:
                part = pf.split('/')[-1][:-5]
                if part in ['body_mask']:
                    continue
                data = json.load(f)
                
                self.contact_verts_ids[part] = list(set(data["verts_ind"]))
                self.contact_faces_ids[part] = list(set(data["faces_ind"]))

    def get_contact_id(self, contact_body_part):
        """ Get contact body part, i.e. vertices ids and faces ids

        Args:
            contact_body_part: contact body part list
        
        Return:
            Contact vertice index and faces index
        """
        verts_ids = []
        faces_ids = []
        for part in contact_body_part:
            #if(self.contact_verts_ids[part] < 6890):
            '''vids=[]
            for i in range(len(self.contact_verts_ids[part])):
                if(self.contact_verts_ids[part][i] < 6890):
                    vids.append(self.contact_verts_ids[part][i])'''
            verts_ids.append(self.contact_verts_ids[part])
            faces_ids.append(self.contact_faces_ids[part])

        verts_ids = np.concatenate(verts_ids)
        faces_ids = np.concatenate(faces_ids)

        return verts_ids, faces_ids

smplx_geometry = SMPLXGeometry("/public/home/wangzy17/Scene-Diffuser/body_segments")
contact_body_part_vid, _ = smplx_geometry.get_contact_id(
    ['back','gluteus','L_Hand','R_Hand','L_Leg','R_Leg','thighs']
)
        
def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles
def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))
def get_marker_indices():
    markers = {
        "gender": "unknown",
        "markersets": [
            {
                "distance_from_skin": 0.0095,
                "indices": {
                    "C7": 3832,
                    "CLAV": 5533,
                    "LANK": 5882,
                    "LFWT": 3486,
                    "LBAK": 3336,
                    "LBCEP": 4029,
                    "LBSH": 4137,
                    "LBUM": 5694,
                    "LBUST": 3228,
                    "LCHEECK": 2081,
                    "LELB": 4302,
                    "LELBIN": 4363,
                    "LFIN": 4788,
                    "LFRM2": 4379,
                    "LFTHI": 3504,
                    "LFTHIIN": 3998,
                    "LHEE": 8846,
                    "LIWR": 4726,
                    "LKNE": 3682,
                    "LKNI": 3688,
                    "LMT1": 5890,
                    "LMT5": 5901,
                    "LNWST": 3260,
                    "LOWR": 4722,
                    "LBWT": 5697,
                    "LRSTBEEF": 5838,
                    "LSHO": 4481,
                    "LTHI": 4088,
                    "LTHMB": 4839,
                    "LTIB": 3745,
                    "LTOE": 5787,
                    "MBLLY": 5942,
                    "RANK": 8576,
                    "RFWT": 6248,
                    "RBAK": 6127,
                    "RBCEP": 6776,
                    "RBSH": 7192,
                    "RBUM": 8388,
                    "RBUSTLO": 8157,
                    "RCHEECK": 8786,
                    "RELB": 7040,
                    "RELBIN": 7099,
                    "RFIN": 7524,
                    "RFRM2": 7115,
                    "RFRM2IN": 7303,
                    "RFTHI": 6265,
                    "RFTHIIN": 6746,
                    "RHEE": 8634,
                    "RKNE": 6443,
                    "RKNI": 6449,
                    "RMT1": 8584,
                    "RMT5": 8595,
                    "RNWST": 6023,
                    "ROWR": 7458,
                    "RBWT": 8391,
                    "RRSTBEEF": 8532,
                    "RSHO": 6627,
                    "RTHI": 6832,
                    "RTHMB": 7575,
                    "RTIB": 6503,
                    "RTOE": 8481,
                    "STRN": 5531,
                    "T8": 5487,
                    "LFHD": 707,
                    "LBHD": 2026,
                    "RFHD": 2198,
                    "RBHD": 3066
                },
                "marker_radius": 0.0095,
                "type": "body"
            }
        ]
    }
    marker_indic = list(markers['markersets'][0]['indices'].values())
    #marker_indic = [index if index < 6890 else 0 for index in marker_indices] ##

    return marker_indic


def smplx_signed_distance(object_points, smplx_vertices, smplx_face):
    """ Compute signed distance between query points and mesh vertices.
    
    Args:
        object_points: (B, O, 3) query points in the mesh.
        smplx_vertices: (B, H, 3) mesh vertices.
        smplx_face: (F, 3) mesh faces.
    
    Return:
        signed_distance_to_human: (B, O) signed distance to human vertex on each object vertex
        closest_human_points: (B, O, 3) closest human vertex to each object vertex
        signed_distance_to_obj: (B, H) signed distance to object vertex on each human vertex
        closest_obj_points: (B, H, 3) closest object vertex to each human vertex
    """
    # compute vertex normals
    smplx_face_vertices = smplx_vertices[:, smplx_face]
    e1 = smplx_face_vertices[:, :, 1] - smplx_face_vertices[:, :, 0]
    e2 = smplx_face_vertices[:, :, 2] - smplx_face_vertices[:, :, 0]
    e1 = e1 / torch.norm(e1, dim=-1, p=2).unsqueeze(-1)
    e2 = e2 / torch.norm(e2, dim=-1, p=2).unsqueeze(-1)
    smplx_face_normal = torch.cross(e1, e2)     # (B, F, 3)

    # compute vertex normal
    smplx_vertex_normals = torch.zeros(smplx_vertices.shape).float()
    smplx_vertex_normals.index_add_(1, smplx_face[:,0], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,1], smplx_face_normal)
    smplx_vertex_normals.index_add_(1, smplx_face[:,2], smplx_face_normal)
    smplx_vertex_normals = smplx_vertex_normals / torch.norm(smplx_vertex_normals, dim=-1, p=2).unsqueeze(-1)

    # compute paired distance of each query point to each face of the mesh
    pairwise_distance = torch.norm(object_points.unsqueeze(2) - smplx_vertices.unsqueeze(1), dim=-1, p=2)    # (B, O, H)
    
    # find the closest face for each query point
    distance_to_human, closest_human_points_idx = pairwise_distance.min(dim=2)  # (B, O)
    closest_human_point = smplx_vertices.gather(1, closest_human_points_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, O, 3)
    query_to_surface = closest_human_point - object_points
    query_to_surface = query_to_surface / torch.norm(query_to_surface, dim=-1, p=2).unsqueeze(-1)
    closest_vertex_normals = smplx_vertex_normals.gather(1, closest_human_points_idx.unsqueeze(-1).repeat(1, 1, 3))
    same_direction = torch.sum(query_to_surface * closest_vertex_normals, dim=-1)
    signed_distance_to_human = same_direction.sign() * distance_to_human    # (B, O)
    
    # find signed distance to object from human
    # signed_distance_to_object = torch.zeros([pairwise_distance.shape[0], pairwise_distance.shape[2]]).float().cuda()-10  # (B, H)
    # signed_distance_to_object, closest_obj_points_idx = torch_scatter.scatter_max(signed_distance_to_human, closest_human_points_idx, out=signed_distance_to_object)
    # closest_obj_points_idx[closest_obj_points_idx == pairwise_distance.shape[1]] = 0
    # closest_object_point = object_points.gather(1, closest_obj_points_idx.unsqueeze(-1).repeat(1,1,3))
    # return signed_distance_to_human, closest_human_point, signed_distance_to_object, closest_object_point, smplx_vertex_normals
    return signed_distance_to_human, closest_human_point

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


if __name__ == '__main__':
    SMPLX = SMPLXWrapper("/public/home/wangzy17/motion-diffusion-model/body_models/smplx_models/", 'cpu', 12)  ####
    nframes=40
    contact_threshold=0.05
    
    '''originmotions=pickle.load(open(pjoin("/storage/group/4dvlab/congpsh/Diff-Motion/","test_split_motion.pkl"),"rb"))  ####originmotions
    index=4
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
    floor_height = positions.min(axis=0).min(axis=0)[1]  #floor height
    print(floor_height)'''
    
    name="test111"
    scene_pos = np.fromfile(pjoin("/public/home/wangzy17/motion-diffusion-model/newscene/" + name + '.bin'), dtype=np.float32).reshape(-1, 6)[:,:3] #pos
    
    smplx_params=np.load(pjoin("/public/home/wangzy17/motion-diffusion-model/scene_generate/"+name+".sample00_rep00_smpl_params.npy"),allow_pickle=True).item()
    
    res = defaultdict(list)
    
    smplx_trans=smplx_params["root_translation"].reshape(nframes,3)  #nframesx3
    smplx_beta=smplx_params["betas"] #nframesx10 
    smplx_pose= smplx_params["thetas"].reshape(nframes,24,6)[:,:22,:] #nframesx22x6
    smplx_pose=np.array(matrix_to_axis_angle(rotation_6d_to_matrix(torch.tensor(smplx_pose))).reshape(40,66)) #smpl model:R+pose 
    smpldict=dict()
    smpldict['transl']=torch.tensor(smplx_trans)
    smpldict['global_orient']=torch.tensor(smplx_pose[:,:3])
    smpldict['body_pose']=torch.tensor(smplx_pose[:,3:])
    smpldict['betas']=torch.tensor(smplx_beta)
    
    ## diversity metrics
    ## 0. transl pairwise distance and standard deviation
    transl_np = smplx_trans  #smpl model :translation
    k, D = transl_np.shape #nframes x 3
    print(k)
    print(D)
    print("0")
    transl_pdist = pairwise_distances(
        transl_np, transl_np, metric='l2').sum() / (k * (k - 1))
    transl_std = np.std(transl_np, axis=0).mean()

    res['transl_pdist'].append(float(transl_pdist))
    res['transl_std'].append(float(transl_std))

    ## 1. smplx parameter pairwise distance and standard deviation
    smplx_params_np = smplx_pose
    k, D = smplx_params_np.shape #nframes x 66
    print(k)
    print(D)
    print("1")
    param_pdist = pairwise_distances(
        smplx_params_np, smplx_params_np, metric='l2').sum() / (k * (k - 1))
    param_std = np.std(smplx_params_np, axis=0).mean()

    res['param_pdist'].append(float(param_pdist))
    res['param_std'].append(float(param_std))

    ## 2. global body marker pairwise distance and standard deviation
    smplx_params_local=copy.deepcopy(smpldict)
    smplx_params_local["transl"]=torch.zeros(40, 3) ###
    body_verts_local, body_faces, body_joints_local =SMPLX.run(smplx_params_local) #####
    #for i in range(body_verts_local.shape[0]):
    #body_verts_local[:,:,1]-=floor_height
    #body_verts_local, body_faces= smplx_params_local["vertices"].reshape(nframes,6890,3),smplx_params_local["faces"] #从npy来
    body_verts_local_np = body_verts_local.cpu().numpy() #nfrmes x len(10475) x 3
    k, V, D = body_verts_local_np.shape
    print(k)
    print(V)
    print(D)
    print("2")
    body_marker = body_verts_local_np.reshape(k, V, D)[:, get_marker_indices(), :]
    body_marker = body_marker.reshape(k, -1) # <k, M * 3>, concatenation of M marker coordinates
    marker_pdist = pairwise_distances(
        body_marker, body_marker, metric='l2').sum() / (k * (k - 1))
    marker_std = np.std(body_marker, axis=0).mean()

    res['marker_pdist'].append(float(marker_pdist))
    res['marker_std'].append(float(marker_std))

    ## physics metrics
    ## non-collision score and contact score
    body_verts_local, body_faces, body_joints_local =SMPLX.run(smpldict) #####
    #body_verts_local[:,:,1]-=floor_height
    non_collision_score = []
    contact_score = []
    scene_verts = np.array(torch.tensor(scene_pos).unsqueeze(0))
    body_verts_tensor = body_verts_local#to device 
    body_faces_tensor = torch.tensor(body_faces.astype(np.int64))
    for j in range(k):
        print(j)
        scene_to_human_sdf, _ = smplx_signed_distance(
            object_points=torch.tensor(scene_verts),
            smplx_vertices=body_verts_tensor[j:j+1],
            smplx_face=body_faces_tensor
        ) # <B, O> = D(<B, O, 3>, <B, H, 3>)
        sdf = scene_to_human_sdf.cpu().numpy() # <1, O>
        non_collision = np.sum(sdf <= 0) / sdf.shape[-1]
        non_collision_score.append(non_collision)
        
        ## computation method of paper "Generating 3D People in Scenes without People"
        ## not very reasonable
        # if np.sum(sdf > 0) > 0:
        #     contact = 1.0
        # else:
        #     contact = 0.0
        # contact_score.append(contact)
        ## we compute the chamfer distance between contact body part and scene
        body_verts_contact = body_verts_tensor[j:j+1][:, contact_body_part_vid, :]
        dist1, dist2, idx1, idx2 = chamfer_dist(
            body_verts_contact,
            torch.tensor(scene_verts)
        )
        print(dist1)
        if torch.sum(dist1 < contact_threshold) > 0:
            contact = 1.0
        else:
            contact = 0.0
        contact_score.append(contact)

    res['non_collision'].append(sum(non_collision_score) / len(non_collision_score))
    res['contact'].append(sum(contact_score) / len(contact_score))

    for key in ['transl_pdist', 'transl_std', 'param_pdist', 'param_std', 'marker_pdist', 'marker_std', 'non_collision', 'contact']:
        res[key+'_average'] = sum(res[key]) / len(res[key])

    save_dir="/public/home/wangzy17/motion-diffusion-model/scene_generate/metrics/"
    save_path = os.path.join(save_dir+ name+ '.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fp:
        json.dump(res, fp)