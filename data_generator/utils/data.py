import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from utils import transform
import pytorch3d as torch3d
import utils
from models.smpl import JOINT_IDS

IMU_VERTEX_IDX = [6723, 3322, 5669, 2244]

KIN_CHAINS = [[5, 2, 0],
              [4, 1, 0],
              [19, 17, 14, 9, 6, 3, 0],
              [18, 16, 13, 9, 6, 3, 0]]


def get_target_view(batch, target_view_batch):
    batch_size = batch.shape[0]
    batch_target_view_list = []
    for i in range(batch_size):
        target_view = target_view_batch[i]
        batch_target_view = batch[i, target_view, ...][None, ...]
        batch_target_view_list.append(batch_target_view)

    batch_target_view = torch.cat(batch_target_view_list, 0)
    return batch_target_view

BONES = [('OP LElbow', 'OP LWrist'),('OP RKnee', 'OP RAnkle'), ('OP LShoulder', 'OP LElbow'),('OP Neck', 'OP LShoulder'),('OP Neck', 'OP Nose'),
              ('OP Neck', 'OP MidHip'),('OP RHip', 'OP RKnee'),('OP RAnkle', 'OP RBigToe')]

def get_bone_length(shape_batch, smpl):# (b, 10)->(b, 8)
    device = shape_batch.device
    batch_size = shape_batch.shape[0]
    output = smpl(betas=shape_batch,
              body_pose=torch.zeros(batch_size, 69).to(device),
              global_orient=torch.zeros(batch_size, 3).to(device),
              transl=torch.zeros(batch_size, 3).to(device), return_vertices=False)
    joints3d = output.joints.reshape(batch_size, 21, 3)
    pair_ids = [(JOINT_IDS[item[0]], JOINT_IDS[item[1]]) for item in BONES]
    bone_length = []
    for pair in pair_ids:
        bone = (joints3d[:,pair[0],:]-joints3d[:,pair[1],:]).norm(dim=-1, keepdim=True)
        bone_length.append(bone)
    bone_length = torch.cat(bone_length, -1)
    return bone_length


def preprocess_simul(batch, smpl, device):

    dof = batch.to(device)
    batch_size = dof.shape[0]
    
    imu, joints2d_target_view, shape, dof = simulator.simulate(dof)

    fullpose_mat = torch3d.transforms.axis_angle_to_matrix(dof[..., 3:].reshape(batch_size, -1, 24, 3))
    fullpose_6d = torch3d.transforms.matrix_to_rotation_6d(fullpose_mat)
    fullpose_9d = fullpose_mat.reshape(batch_size, -1, 24, 9)

    r2c_9d = fullpose_9d[:, :, :1, :]
    pose_9d = fullpose_9d[:, :, 1:, :]
    transl = dof[:, :, :3]

    # 2D Joints
    joints_velo2d = joints2d_target_view[:, 1:, :, :2] - joints2d_target_view[:, :-1, :, :2]
    joints2d_norm = joints2d_target_view[:, :, :, :2] - joints2d_target_view[:, :, 11:12, :2]
    joints2d_info = torch.cat([joints2d_norm[:, 1:, :, :], joints_velo2d, joints2d_target_view[:, 1:, :, 2:3]], 3)
    joints2d_info = joints2d_info.reshape(batch_size, -1, 105)

    # 3D Joints
    seq_len = pose_9d.shape[1]
    pose_blob = pose_9d.reshape(-1, 23, 9)
    shape_blob = shape[:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10)
    r2c_blob = r2c_9d.reshape(-1, 1, 9)
    transl_blob = transl.reshape(-1, 3)
    output = smpl(betas=shape_blob, body_pose=pose_blob, global_orient=r2c_blob, transl=transl_blob,
                  return_vertices=False, pose2rot=False)
    joints3d = output.joints.reshape(batch_size, -1, 21, 3)

    b2c_list = [imu[:, 1:, :9].reshape(batch_size, -1, 3, 3), imu[:, 1:, 12:21].reshape(batch_size, -1, 3, 3),
                imu[:, 1:, 24:33].reshape(batch_size, -1, 3, 3), imu[:, 1:, 36:45].reshape(batch_size, -1, 3, 3)]


    b2c_list = transform.get_b2c(fullpose_9d)
    bone_length = get_bone_length(shape, smpl)
    input_info = torch.cat(b2c_list+[imu[:, :, 9:12], imu[:, :, 21:24], imu[:,:, 33:36], imu[:, :, 45:48], r2c_9d[...,0,:], bone_length[:, None, :].repeat([1, seq_len, 1])], 2)
    
    data = {}
    data["input_info"] = input_info
    data["shape"] = shape
    data["bone_length"] = bone_length
    data["fullpose_9d"] = fullpose_9d
    data["fullpose_6d"] = fullpose_6d

    data["r2c_9d"] = r2c_9d
    data["transl"] = transl

    data["joints3d"] = joints3d
    data["b2c_list"] = b2c_list

    return data



def preprocess_v2(batch, smpl, device):
    Ri2w = batch['Ri2w'].to(device)
    Rs2b = batch['Rs2b'].to(device)
    imu = batch['imu'].to(device)

    dof_world = batch['gt_transpose'].to(device)
    shape = batch['gt_shape'].to(device)

    Gw2c_allview = batch['Gw2c_allview'].to(device)
    proj_allview = batch['proj_allview'].to(device)
    joints2d_allview = batch['joints2d_allview'].to(device)

    key = batch['key']
    view = batch['view']
    interval = batch['interval']
    first_cam_frame = batch['first_cam_frame']

    batch_size = dof_world.shape[0]

    Gw2c_target_view = get_target_view(Gw2c_allview, view)
    joints2d_target_view = get_target_view(joints2d_allview, view)
    proj_target_view = get_target_view(proj_allview, view)
    #     print(Gw2c_allview.shape)
    #     print(proj_allview.shape)
    #     print(joints2d_allview.shape)
    #     print(Gw2c_target_view.shape)
    #     print(joints2d_target_view.shape)

    Rw2c = Gw2c_target_view[:, :3, :3]
    Ri2c = []
    for idx in range(Rw2c.shape[0]):
        rot = Rw2c[idx].mm(Ri2w[idx])
        Ri2c.append(rot[None, :, :])
    Ri2c = torch.cat(Ri2c, 0)

    r2w_3d = dof_world[:, :, 3:6].clone()
    transl_world = dof_world[:, :, :3].clone()
    imu = transform.transform_imu_v2(Ri2c, Rs2b, imu)
    dof = dof_world.clone()
    dof[:, :, :6] = transform.transform_frame_v2(Gw2c_target_view, dof[:, :, :6], shape, smpl)

    fullpose_mat = torch3d.transforms.axis_angle_to_matrix(dof[..., 3:].reshape(batch_size, -1, 24, 3))
    fullpose_6d = torch3d.transforms.matrix_to_rotation_6d(fullpose_mat)
    fullpose_9d = fullpose_mat.reshape(batch_size, -1, 24, 9)

    r2c_9d = fullpose_9d[:, :, :1, :]
    pose_9d = fullpose_9d[:, :, 1:, :]
    transl = dof[:, :, :3]


    # 2D Joints
    joints2d_norm = joints2d_target_view.clone()

    # translationXY
    delta_xy=np.zeros(joints2d_norm.shape,dtype=np.float32)
    delta_uv=torch.from_numpy(delta_xy).to(device)
    for s in range(delta_xy.shape[0]): 
        delta_xy[s,:,:,0]=np.random.uniform(-1*torch.min(joints2d_norm[...,0]).cpu().numpy(),4096-torch.max(joints2d_norm[...,0]).cpu().numpy())
        delta_xy[s,:,:,1]=np.random.uniform(-1*torch.min(joints2d_norm[...,1]).cpu().numpy(),2160-torch.max(joints2d_norm[...,1]).cpu().numpy())
    delta_xy = torch.from_numpy(delta_xy).to(device)
    joints2d_norm=joints2d_norm+delta_xy
    
    delta_uv[:, :, :, 0] = delta_xy[:, :, :, 0] / proj_target_view[:,0,0][:,None,None]
    delta_uv[:, :, :, 1] = delta_xy[:, :, :, 1] / proj_target_view[:,1,1][:,None,None]
    
    delta_XY=torch.zeros(transl.shape,dtype=torch.float32).to(device)
    
    delta_XY[:,:,0]=transl[:,:,2]*delta_uv[:,:,0,0]
    delta_XY[:,:,1]=transl[:,:,2]*delta_uv[:,:,0,1]
    transl=transl+delta_XY
    
    root2d = transform.batch_proj(transl[...,None,:], proj_allview)
    root2d = get_target_view(root2d, view)[:,:,0,:]
    #print(root2d.shape)
    
    
    k_z=torch.zeros(joints2d_norm.shape[0],dtype=torch.float32).to(device)
    for s in range(delta_xy.shape[0]):
        root2d_cur = root2d[s, :, :]
        max_x=torch.max(joints2d_norm[s, :, :, 0])
        min_x=torch.min(joints2d_norm[s, :, :, 0])
        max_y=torch.max(joints2d_norm[s, :, :, 1])
        min_y=torch.min(joints2d_norm[s, :, :, 1])
        upper_bound=min(torch.min((4096-root2d_cur[:,0])/(max_x-root2d_cur[:,0])).item(),torch.min(root2d_cur[:,0]/(root2d_cur[:,0]-min_x)).item(),torch.min((2160-root2d_cur[:,1])/(max_y-root2d_cur[:,1])).item(),torch.min(root2d_cur[:,1]/(root2d_cur[:,1]-min_y)).item())
        lower_bound=0.5
        k_z[s]=torch.rand(1)*(upper_bound-lower_bound)+lower_bound
        joints2d_norm[s,:,:,0]=root2d_cur[:,0][:,None]+k_z[s]*(joints2d_norm[s,:,:,0]-root2d_cur[:,0][:,None])
        joints2d_norm[s,:,:,1]=root2d_cur[:,1][:,None]+k_z[s]*(joints2d_norm[s,:,:,1]-root2d_cur[:,1][:,None])
#         print(root2d_cur.shape)
#         print(joints2d_norm.shape)
        
    

    transl[:,:,2]=transl[:,:,2]/k_z[:,None]
    #print(joints2d_norm.shape)
    
# reflection
    transl[:,:,0]=-transl[:,:,0]
    for i,j in [(0,1),(2,3),(4,5),(6,7),(8,9),(10,12),(15,16),(17,18),(19,20)]:
        transl[:,i,:],transl[:,j,:]=transl[:,j,:],transl[:,i,:]
    root2d = transform.batch_proj(transl[...,None,:], proj_allview)
    root2d = get_target_view(root2d, view)[:,:,0,:]
    #print(root2d.shape)
    for s in range(joints2d_norm.shape[0]):
        joints2d_norm[s,:,:,:]=root2d[s,:,:][:,None]
    
    
    joints2d_norm[:, :, :, 0] = (joints2d_norm[:, :, :, 0] - proj_target_view[:,0,2][:,None,None]) / proj_target_view[:,0,0][:,None,None] #X
    joints2d_norm[:, :, :, 1] = (joints2d_norm[:, :, :, 1] - proj_target_view[:,1,2][:,None,None]) / proj_target_view[:,1,1][:,None,None] #Y
    #print(joints2d_norm.shape)

    joints2d_info = joints2d_norm.reshape(batch_size, -1, 63)
#     root2d[:, :, 0] = (root2d[:, :, 0] - proj_target_view[:,0,2][:,None]) / proj_target_view[:,0,0][:,None]
#     root2d[:, :, 1] = (root2d[:, :, 1] - proj_target_view[:,1,2][:,None]) / proj_target_view[:,1,1][:,None]
#     root2d = transform.batch_proj(transl[...,None,:], proj_allview)
#     root2d = get_target_view(root2d, view)[:,:,0,:]

    # 3D Joints
    seq_len = pose_9d.shape[1]
    pose_blob = pose_9d.reshape(-1, 23, 9)
    shape_blob = shape[:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10)
    r2c_blob = r2c_9d.reshape(-1, 1, 9)
    transl_blob = transl.reshape(-1, 3)
    output = smpl(betas=shape_blob, body_pose=pose_blob, global_orient=r2c_blob, transl=transl_blob,
                  return_vertices=False, pose2rot=False)
    joints3d = output.joints.reshape(batch_size, -1, 21, 3)
    
    


    # input_info = torch.cat([imu, joint2d], 2)
    
    # init_info = torch.cat([imu[:,0,:], joint2d[:,0,:], joint3d[:,0,:]], 1)
    b2c_list = [imu[:, 1:, :9].reshape(batch_size, -1, 3, 3), imu[:, 1:, 12:21].reshape(batch_size, -1, 3, 3),
                imu[:, 1:, 24:33].reshape(batch_size, -1, 3, 3), imu[:, 1:, 36:45].reshape(batch_size, -1, 3, 3)]

    #input_info = torch.cat([imu[:, 1:, :], joints2d_info, shape[:, None, :].repeat([1, joints2d_info.shape[1], 1])], 2)
    b2c_list = transform.get_b2c(fullpose_9d)
    bone_length = get_bone_length(shape, smpl)
    input_info = torch.cat(b2c_list+[imu[:, :, 9:12], imu[:, :, 21:24], imu[:,:, 33:36], imu[:, :, 45:48], joints2d_info, bone_length[:, None, :].repeat([1, seq_len, 1])], 2)
    
    data = {}
    data["input_info"] = input_info
    data["shape"] = shape
    data["bone_length"] = bone_length
    data["fullpose_9d"] = fullpose_9d
    data["fullpose_6d"] = fullpose_6d

    data["Gw2c_allview"] = Gw2c_allview
    data["proj_allview"] = proj_allview
    data["joints2d_allview"] = joints2d_allview

    data["key"] = key
    data["view"] = view
    data["interval"] = interval
    data["first_cam_frame"] = first_cam_frame

    data["r2w_3d"] = r2w_3d
    data["transl_world"] = transl_world

    data["r2c_9d"] = r2c_9d
    data["transl"] = transl

    data["joints3d"] = joints3d
    data["b2c_list"] = b2c_list

    return data


def postprocess_v2(net_output, data, smpl):
    batch_size, seq_len, _ = net_output['p_limb'].shape
    
    fullpose_6d_pred = net_output['fullpose_6d'].view(batch_size, -1, 24, 6)
    fullpose_9d_pred = torch3d.transforms.rotation_6d_to_matrix(fullpose_6d_pred).view(batch_size, -1, 24, 9)
    
    pose_6d_pred = fullpose_6d_pred[..., 1:, :]
    r2c_6d_net_pred = fullpose_6d_pred[..., :1, :]
    pose_9d_pred = fullpose_9d_pred[..., 1:, :]
    r2c_9d_net_pred = fullpose_9d_pred[..., :1, :]

    
    # -------------------------- FK -----------------------------------------
    shape_blob = data['shape'][:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10)
    pose_blob = pose_9d_pred.reshape(-1, 23, 9)
    r2c_blob = r2c_9d_net_pred.reshape(-1, 1, 9)
    transl_blob = data["transl"].reshape(-1, 3)
    pred = smpl(betas=shape_blob, body_pose=pose_blob, global_orient=r2c_blob, transl=transl_blob,
                return_vertices=False,
                pose2rot=False)
    joints3d_pred = pred.joints.view(batch_size, -1, 21, 3)
    
    

    # -------------------------- SLERP -----------------------------------------
    
    b2c_list = transform.get_b2c(fullpose_9d_pred)
    b2r_list = transform.get_b2r(fullpose_9d_pred)
    r2c_list = []
    for idx, b2r in enumerate(b2r_list):
        r2c = torch.einsum("btmn,btnl->btml", [data["b2c_list"][idx].view(batch_size, seq_len, 3, 3), b2r.view(batch_size, seq_len, 3, 3).permute([0,1,3,2])])
        r2c_list.append(r2c)

    q0 = torch3d.transforms.matrix_to_quaternion(r2c_list[0])
    q1 = torch3d.transforms.matrix_to_quaternion(r2c_list[1])
    q2 = torch3d.transforms.matrix_to_quaternion(r2c_list[2])
    q3 = torch3d.transforms.matrix_to_quaternion(r2c_list[3])

    q_first = transform.batch_slerp(q0, q2)
    q_second = transform.batch_slerp(q1, q3)

    q = transform.batch_slerp(q_first, q_second)

    q_net = torch3d.transforms.matrix_to_quaternion(r2c_9d_net_pred.view(batch_size, -1, 3, 3))
    q = transform.batch_slerp(q, q_net)

    r2c_9d_final_pred = torch3d.transforms.quaternion_to_matrix(q).view(batch_size, -1, 1, 9)
    
    # -------------------------- PROJECTION -----------------------------------------
    joints2d_pred_allview = transform.batch_proj(joints3d_pred, data["proj_allview"])

    joints3d_vel_pred = joints3d_pred[:, 1:, ...] - joints3d_pred[:, :-1, ...]
    joints3d_acc_pred = (joints3d_vel_pred[:, 1:, ...] - joints3d_vel_pred[:, :-1, ...]).norm(dim=3)
    joints3d_vel_pred = joints3d_vel_pred.norm(dim=3)


    #dof_world_pred = torch.cat(
     #   [data["transl_world"], data["r2w_3d"], pose_3d_pred], 2)
    

    data = {}

    data['p_limb'] = net_output['p_limb'].view(batch_size, seq_len, -1, 3)
    data['p_body'] = net_output['p_body'].view(batch_size, seq_len, -1, 3)
    
    data['joints3d'] = joints3d_pred
    data['joints3d_vel'] = joints3d_vel_pred
    data['joints3d_acc'] = joints3d_acc_pred

    # data['joints2d_pred_list'] = joints2d_pred_list

    #data['dof_world'] = dof_world_pred

    data['fullpose_9d'] = fullpose_9d_pred
    data['r2c_9d_net'] = r2c_9d_net_pred
    data['r2c_9d_final'] = r2c_9d_final_pred

    data['fullpose_6d'] = fullpose_6d_pred
    data['r2c_6d_net'] = r2c_6d_net_pred

    data["joints2d_allview"] = joints2d_pred_allview
    
    data["r2c_mat_list"] = r2c_list
    data["b2c_list"] = b2c_list
    
    return data


def postprocess(net_output, data, smpl):
    batch_size, seq_len, _ = net_output['fullpose_6d'].shape
    fullpose_6d_pred = net_output['fullpose_6d'].view(batch_size, -1, 24, 6)
    fullpose_9d_pred = torch3d.transforms.rotation_6d_to_matrix(fullpose_6d_pred).view(batch_size, -1, 24, 9)
    fullpose_3d_pred = torch3d.transforms.so3_log_map(fullpose_9d_pred.view(-1, 3, 3)).view(batch_size, -1, 24 * 3)
    
    pose_6d_pred = fullpose_6d_pred[..., 1:, :]
    r2c_6d_net_pred = fullpose_6d_pred[..., :1, :]
    pose_9d_pred = fullpose_9d_pred[..., 1:, :]
    r2c_9d_net_pred = fullpose_9d_pred[..., :1, :]
    pose_3d_pred = fullpose_3d_pred[..., 3:]
    r2c_3d_net_pred = fullpose_3d_pred[..., :3]
    
    b2c_list = transform.get_b2c(fullpose_9d_pred)
    b2r_list = transform.get_b2r(fullpose_9d_pred)
    r2c_list = []
    for idx, b2r in enumerate(b2r_list):
        r2c = torch.einsum("btmn,btnl->btml", [data["b2c_list"][idx], b2r.permute([0,1,3,2])])
        r2c_list.append(r2c)

    q0 = torch3d.transforms.matrix_to_quaternion(r2c_list[0])
    q1 = torch3d.transforms.matrix_to_quaternion(r2c_list[1])
    q2 = torch3d.transforms.matrix_to_quaternion(r2c_list[2])
    q3 = torch3d.transforms.matrix_to_quaternion(r2c_list[3])

    q_first = transform.batch_slerp(q0, q2)
    q_second = transform.batch_slerp(q1, q3)

    q = transform.batch_slerp(q_first, q_second)

    q_net = torch3d.transforms.matrix_to_quaternion(r2c_9d_net_pred.view(batch_size, -1, 3, 3))
    q = transform.batch_slerp(q, q_net)

    r2c_9d_final_pred = torch3d.transforms.quaternion_to_matrix(q).view(batch_size, -1, 1, 9)
    

    shape_blob = data['shape'][:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10)
    pose_blob = pose_9d_pred.reshape(-1, 23, 9)
    r2c_blob = r2c_9d_net_pred.reshape(-1, 1, 9)
    transl_blob = data["transl"][:, 1:, :].reshape(-1, 3)
    pred = smpl(betas=shape_blob, body_pose=pose_blob, global_orient=r2c_blob, transl=transl_blob,
                return_vertices=False,
                pose2rot=False)

    joints3d_pred = pred.joints.view(batch_size, -1, 21, 3)
    # for i in range():

    # joints2d_pred_allview = transform.batch_proj(joints3d_pred, utils.data.get_target_view(data["proj_allview"], data["view"]))
    joints2d_pred_allview = transform.batch_proj(joints3d_pred, data["proj_allview"])

    joints3d_vel_pred = joints3d_pred[:, 1:, ...] - joints3d_pred[:, :-1, ...]
    joints3d_acc_pred = (joints3d_vel_pred[:, 1:, ...] - joints3d_vel_pred[:, :-1, ...]).norm(dim=3)
    joints3d_vel_pred = joints3d_vel_pred.norm(dim=3)


    dof_world_pred = torch.cat(
        [data["transl_world"][:, 1:, :], data["r2w_3d"][:, 1:, :], pose_3d_pred], 2)
    
    

    

    data = {}

    data['joints3d'] = joints3d_pred
    data['joints3d_vel'] = joints3d_vel_pred
    data['joints3d_acc'] = joints3d_acc_pred

    # data['joints2d_pred_list'] = joints2d_pred_list

    data['dof_world'] = dof_world_pred

    data['fullpose_9d'] = fullpose_9d_pred
    data['r2c_9d_net'] = r2c_9d_net_pred
    data['r2c_9d_final'] = r2c_9d_final_pred

    data['fullpose_6d'] = fullpose_6d_pred
    data['r2c_6d_net'] = r2c_6d_net_pred

    data["joints2d_allview"] = joints2d_pred_allview
    
    data["r2c_mat_list"] = r2c_list
    data["b2c_list"] = b2c_list
    return data


def preprocess(batch, smpl, device):
    Ri2w = batch['Ri2w'].to(device)
    Rs2b = batch['Rs2b'].to(device)
    imu = batch['imu'].to(device)

    dof_world = batch['gt_transpose'].to(device)
    shape = batch['gt_shape'].to(device)

    Gw2c_allview = batch['Gw2c_allview'].to(device)
    proj_allview = batch['proj_allview'].to(device)
    joints2d_allview = batch['joints2d_allview'].to(device)

    key = batch['key']
    view = batch['view']
    interval = batch['interval']
    first_cam_frame = batch['first_cam_frame']

    batch_size = dof_world.shape[0]

    Gw2c_target_view = get_target_view(Gw2c_allview, view)
    joints2d_target_view = get_target_view(joints2d_allview, view)

    Rw2c = Gw2c_target_view[:, :3, :3]
    Ri2c = []
    for idx in range(Rw2c.shape[0]):
        rot = Rw2c[idx].mm(Ri2w[idx])
        Ri2c.append(rot[None, :, :])
    Ri2c = torch.cat(Ri2c, 0)

    r2w_3d = dof_world[:, :, 3:6].clone()
    transl_world = dof_world[:, :, :3].clone()
    imu = transform.transform_imu_v2(Ri2c, Rs2b, imu)
    dof = dof_world.clone()
    dof[:, :, :6] = transform.transform_frame_v2(Gw2c_target_view, dof[:, :, :6], shape, smpl)

    fullpose_mat = torch3d.transforms.axis_angle_to_matrix(dof[..., 3:].reshape(batch_size, -1, 24, 3))
    fullpose_6d = torch3d.transforms.matrix_to_rotation_6d(fullpose_mat)
    fullpose_9d = fullpose_mat.reshape(batch_size, -1, 24, 9)

    r2c_9d = fullpose_9d[:, :, :1, :]
    pose_9d = fullpose_9d[:, :, 1:, :]
    transl = dof[:, :, :3]

    # 2D Joints
    joints_velo2d = joints2d_target_view[:, 1:, :, :2] - joints2d_target_view[:, :-1, :, :2]
    joints2d_norm = joints2d_target_view[:, :, :, :2] - joints2d_target_view[:, :, 11:12, :2]
    joints2d_info = torch.cat([joints2d_norm[:, 1:, :, :], joints_velo2d, joints2d_target_view[:, 1:, :, 2:3]], 3)
    joints2d_info = joints2d_info.reshape(batch_size, -1, 105)

    # 3D Joints
    seq_len = pose_9d.shape[1]
    pose_blob = pose_9d.reshape(-1, 23, 9)
    shape_blob = shape[:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10)
    r2c_blob = r2c_9d.reshape(-1, 1, 9)
    transl_blob = transl.reshape(-1, 3)
    output = smpl(betas=shape_blob, body_pose=pose_blob, global_orient=r2c_blob, transl=transl_blob,
                  return_vertices=False, pose2rot=False)
    joints3d = output.joints.reshape(batch_size, -1, 21, 3)

    # input_info = torch.cat([imu, joint2d], 2)
    input_info = torch.cat([imu[:, 1:, :], joints2d_info, shape[:, None, :].repeat([1, joints2d_info.shape[1], 1])], 2)
    # init_info = torch.cat([imu[:,0,:], joint2d[:,0,:], joint3d[:,0,:]], 1)
    b2c_list = [imu[:, 1:, :9].reshape(batch_size, -1, 3, 3), imu[:, 1:, 12:21].reshape(batch_size, -1, 3, 3),
                imu[:, 1:, 24:33].reshape(batch_size, -1, 3, 3), imu[:, 1:, 36:45].reshape(batch_size, -1, 3, 3)]

    data = {}
    data["input_info"] = input_info
    data["shape"] = shape
    data["fullpose_9d"] = fullpose_9d
    data["fullpose_6d"] = fullpose_6d

    data["Gw2c_allview"] = Gw2c_allview
    data["proj_allview"] = proj_allview
    data["joints2d_allview"] = joints2d_allview
    data["proj_allview"] = proj_allview

    data["key"] = key
    data["view"] = view
    data["interval"] = interval
    data["first_cam_frame"] = first_cam_frame

    data["r2w_3d"] = r2w_3d
    data["transl_world"] = transl_world

    data["r2c_9d"] = r2c_9d
    data["transl"] = transl

    data["joints3d"] = joints3d
    data["b2c_list"] = b2c_list

    return data






class Simulator:
    def __init__(self, smpl, device):
        self.smpl = smpl
        self.device = device
        self.K = torch.Tensor(
            [3084.093017578125, 0.0, 2032.12744140625, 0.0, 3085.515380859375, 1114.2928466796875, 0.0, 0.0,
             1.0]).reshape(3, 3).to(self.device)

    def simulate(self, motion_batch):  # 16x100x75
        batch_size, seq_len, _ = motion_batch.shape

        motion_batch = motion_batch.to(self.device)
        motion_batch[..., :3] = motion_batch[..., :3] - motion_batch[:, 0, :3][:, None, :]

        distance = torch.rand([batch_size, 1], device=self.device) * 4. + 2.
        motion_batch[..., 2] = motion_batch[..., 2] + distance

        motions = motion_batch.reshape(-1, 75)  # 1600x75

        Rw2c = torch.from_numpy(R.random(num=batch_size).as_matrix()).to(self.device)  # 16x3x3

        Rr2w = torch.from_numpy(R.from_rotvec(motions[:, 3:6].cpu()).as_matrix()).to(self.device).reshape(batch_size,
                                                                                                          seq_len, 3,
                                                                                                          3)  # 16x100x3x3

        Rr2c = torch.einsum("bmn,bsnl->bsml", Rw2c, Rr2w).reshape(-1, 3, 3)
        Wr2c = torch.from_numpy(R.from_matrix(Rr2c.cpu()).as_rotvec()).to(self.device).reshape(batch_size, seq_len,
                                                                                               3)  # 16x100x3

        motion_batch[..., 3:6] = Wr2c
        motions[..., 3:6] = Wr2c.reshape(-1, 3)

        shape_batch = torch.normal(mean=torch.zeros(batch_size, 10), std=torch.ones(batch_size, 10) / 2).to(self.device)

        output = self.smpl(betas=shape_batch[:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10),
                           body_pose=motions[:, 6:], global_orient=motions[:, 3:6],
                           transl=motions[:, :3], return_vertices=True)

        imu_position = output.vertices[:, IMU_VERTEX_IDX, :].reshape(batch_size, seq_len, 4, 3)
        velo_batch = (imu_position[:, 1:, :] - imu_position[:, :-1, :]) * 60.  # 16x99x4x3
        accel_batch = (velo_batch[:, 1:, :] - velo_batch[:, :-1, :]) * 60.  # 16x98x4x3
        Rb2c = self.get_Rb2c(motion_batch)
        imu = []
        for i in range(accel_batch.shape[2]):
            imu.append(Rb2c[i][:, 1:-1, :])
            imu.append(accel_batch[..., i, :])
        imu = torch.cat(imu, 2)

        joint3d = output.joints  # 1600x21x3

        joint_proj = torch.einsum("mn,bnj->bmj", self.K, joint3d.permute([0, 2, 1])).permute([0, 2, 1])  # 1600x21x3
        joint_proj /= joint_proj[:, :, 2:3]
        joint_proj = joint_proj.reshape(batch_size, seq_len, 21, 3)
        # joint_velo2d = joint_proj[:, 1:, :, :] - joint_proj[:, :-1, :, :]
        # joint_proj -= joint_proj[:, :, 11:12, :]
        # joint2d = torch.cat([joint_proj[:, 1:-1, :, :], joint_velo2d[:, 1:, :, :]], -1)

        return imu, joint_proj[:, 1:-1, :, :], shape_batch, motion_batch[:, 1:-1, :]

    def get_Rb2c(self, motion_batch):
        batch_size, seq_len, _ = motion_batch.shape

        Rb2c = []
        motions = motion_batch.reshape(-1, 75)
        for chain in KIN_CHAINS:

            rotvec = motions[:, chain[0] * 3 + 3:chain[0] * 3 + 6]
            rot_total = torch.from_numpy(R.from_rotvec(rotvec.cpu()).as_matrix().astype(np.float32)).to(
                self.device)  # 1600x3x3
            for j in chain[1:]:
                rotvec = motions[:, j * 3 + 3: j * 3 + 6]
                rot = torch.from_numpy(R.from_rotvec(rotvec.cpu()).as_matrix().astype(np.float32)).to(
                    self.device)  # 1600x3x3
                rot_total = torch.einsum("bmn,bnl->bml", rot, rot_total)
            Rb2c.append(rot_total.reshape(batch_size, seq_len, 9))
        return Rb2c

