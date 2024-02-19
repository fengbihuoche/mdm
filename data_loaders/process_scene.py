from os.path import join as pjoin

from common.skeleton import Skeleton
import numpy as np
import os
from common.quaternion import *
from humanml.utils.paramUtil import *

import torch
from tqdm import tqdm
import os

import numpy as np
import sys
import os  
from os.path import join as pjoin
import codecs as cs


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def mean_variance(data_dir, save_dir, joints_num):
    file_list = os.listdir(data_dir)
    data_list = []

    for file in file_list:
        data = np.load(pjoin(data_dir, file))
        # print(data.shape)
        if np.isnan(data).any():
            # print(file)
            continue
        if data.shape[0] == 263:
            data = data.reshape(1,263)
            
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    # print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]

    np.save(pjoin(save_dir, 'mean.npy'), Mean)
    np.save(pjoin(save_dir, 'std.npy'), Std)

    return Mean, Std

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

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
def process_file(positions,scene, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]
    # print("positions======+",positions[0],positions.shape)
    # np.savetxt('temp_ori.txt', positions[0], delimiter="\t")
    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)
    #print(positions.shape) 40x22x3
    
    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz
    #print(positions.shape)

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    # print(across)

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    
    print(forward_init.shape)
    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    #scene_init=qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
    #scene_rot=np.ones(pointcloud[:,:3].shape[:-1] + (4,)) * scene_init
    #print(scene_rot.shape
    

    ### roration operation
    # print("rota",root_quat_init,root_quat_init.shape,root_quat_init[0]) #(34, 22, 4)
    positions_b = positions.copy()
    positions = qrot_np(root_quat_init, positions)   
    print(floor_height)
    # scene
    '''Scene process'''
    scene[:,1] -= floor_height
    rotation_quat_scene = np.ones(scene.shape[:-1] + (4,)) * root_quat_init[0,0]
    scene[:,:3] = qrot_np(rotation_quat_scene, scene[:,:3])
    np.savetxt('scene_after.txt', scene[:,:3], delimiter=",")
    print("----scene done")

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()  #still nx3，after tran+rotate
    point=scene.copy() ###scene
    # print("global_positions====",global_positions[0],global_positions.shape)
    # np.savetxt('temp_after.txt', global_positions[0], delimiter="\t")

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)
    
    np.savetxt('temp_after_R.txt', positions[0], delimiter=",")

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity,point

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

theta = -np.pi / 2
rotation_matrix = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
])

split = 'train'
dataset_name = 'LaserHuman'
if dataset_name == 'LaserHuman':
    root_path = '/remote-home/share/Diff-Motion/MotionGPT/'
    anno_path = f'/remote-home/share/Diff-Motion/{split}_split_motion2.pkl'
elif dataset_name == 'hps':
    root_path = '/remote-home/share/Diff-Motion/HPS/'
    anno_path = f'/remote-home/share/Diff-Motion/{split}_split_motion.pkl'
# anno_path = '/remote-home/share/Diff-Motion/HPS/hps-final.pkl'
import pickle
#anno = np.load(anno_path,allow_pickle=True)

# print(anno[0].keys(),anno[0]['newsmpl'][0])
split_file="/storage/group/4dvlab/congpsh/Diff-Motion/MotionGPT/train.txt"

id_list = []
with cs.open(split_file, 'r') as f:
    for line in f.readlines():
        id_list.append(line.strip())
id_list=id_list[2292:2293]
originmotions=pickle.load(open(pjoin("/storage/group/4dvlab/congpsh/Diff-Motion/",split+"_split_motion.pkl"),"rb"))  ####originmotions
index=2292
for name in tqdm(id_list):
#for each in tqdm(anno):
    '''if 'newsmpl' in each.keys():
        joint = each['joints'] + each['newsmpl'][0][:,np.newaxis,:]
    else:
        joint = each['joints']'''
    print(index)
    joint=originmotions[index]["joints"]
    joint = joint - joint[0,0]
    
    pointcloud = np.fromfile(pjoin("/storage/group/4dvlab/congpsh/Diff-Motion/scenes/" + name + '.bin'), dtype=np.float32).reshape(-1, 6)
    trans = originmotions[index]['smpl'][0][0]
    pointcloud[:,:3]= pointcloud[:,:3]-trans-joint[0,0]
    
    rotated_joint = joint[:,:22] @ rotation_matrix.T  #z>>y
    pointcloud[:,:3]=pointcloud[:,:3]@ rotation_matrix.T
    
    data, ground_positions, positions, l_velocity ,point= process_file(rotated_joint, pointcloud,0.002) ##
    point.tofile(pjoin("/public/home/wangzy17/motion-diffusion-model/newscene/"+name+'.bin')) #####
    
    if data.shape[0] == 263:
        data = data.reshape(1,-1)
    #each['para_new'] = data
    rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), 22)
    #print(rec_ric_data.shape)
    np.savetxt('/public/home/wangzy17/motion-diffusion-model/data_loaders/temp_after_R2.txt', rec_ric_data[0,0], delimiter=",")
    index+=1
    print("=========done")

    

'''with open(anno_path, "wb") as f:
    pickle.dump(anno, f)

data_dir = root_path + 'new_joint_vecs'
mean, std = mean_variance(data_dir, data_dir, 22)

if False:
    anno = np.load(anno_path,allow_pickle=True)
    import json
    with open('/remote-home/congpsh/diffusion/MotionGPT/prepare/instructions/template_instructions.json','r',encoding='utf-8')as f:
        sample = json.load(f)
    sample_prompt = sample['Motion-to-Text']['caption']['input']

    data_dir = root_path + 'new_joint_vecs'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # write to txt for text-to-motion  
    l_txt =  []
    split_txt = []
    for i in tqdm(range(len(anno))):
        if i != len(anno)-1:
            # print(info[i]['name'],info[i]['description'])
            id_curr = anno[i]['dense']['dense_pc'].split('.')[0].split('/')[-1]
            id_next = anno[i+1]['dense']['dense_pc'].split('.')[0].split('/')[-1]
            if id_curr == id_next:
                continue
        
        path = f'{root_path}new_joint_vecs/{split}{i}.npy'
        np.save(path,anno[i]['para_new'])
        N = len(sample_prompt)  - 1
        random_id = np.random.choice(N)
        l_txt.append(f'{sample_prompt[random_id]}#{path}\n'.replace('<Motion_Placeholder>','<motion>'))
        split_txt.append(f'{path}\n')

    mean, std = mean_variance(data_dir, data_dir, 22)

    with open("/remote-home/congpsh/diffusion/MotionGPT/demos/m2t_own.txt", "w") as f:
        f.writelines(l_txt)
        
    with open(f'{root_path}{split}.txt', "w") as f:
        f.writelines(split_txt)'''
