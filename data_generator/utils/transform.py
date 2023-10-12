from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
from pytorch3d import transforms

KIN_CHAINS = [[5, 2, 0],
              [4, 1, 0],
              [19, 17, 14, 9, 6, 3, 0],
              [18, 16, 13, 9, 6, 3, 0]]


# def transform_frame(G_batch, ksi_batch, device):# 16x4x4 16x100x6
#     batch_size = ksi_batch.shape[0]
#     seq_len = ksi_batch.shape[1]
#     ksi_new_list = []
#     for i in range(batch_size):
#         G = G_batch[i]
#         rot = torch.from_numpy(R.from_rotvec(ksi_batch[i, :, 3:].cpu()).as_matrix().astype(np.float32)).to(device)
#         trans = ksi_batch[i, :, :3][:,:,None]
#         G_origin = torch.cat([rot, trans], 2)
#         homo = torch.Tensor([[[0,0,0,1]]]*seq_len).to(device)
#         G_origin = torch.cat([G_origin, homo], 1)
#
#         G_new = G.mm(G_origin.permute([1,0,2]).reshape(4, -1))
#         G_new = G_new.reshape(4, seq_len, 4)
#         G_new = G_new.permute([1,0,2])
#
#         #print(G_new[0] - G.mm(G_origin[0]))
#
#         rotvec = torch.from_numpy(R.from_matrix(G_new[:, :3, :3].cpu()).as_rotvec().astype(np.float32)).to(device)
#         trans = G_new[:, :3, -1]
#         ksi_new = torch.cat([trans, rotvec], 1)
#         ksi_new_list.append(ksi_new[None,:,:])
#     return torch.cat(ksi_new_list, 0)


def transform_frame_v2(G_batch, ksi_batch, shape, smpl):  # 16x4x4 16x100x6

    device = ksi_batch.device
    batch_size = ksi_batch.shape[0]
    seq_len = ksi_batch.shape[1]

    output = smpl(betas=shape[:, None, :].repeat([1, seq_len, 1]).reshape(-1, 10),
                  body_pose=torch.zeros(batch_size * seq_len, 69).to(device),
                  global_orient=torch.zeros(batch_size * seq_len, 3).to(device),
                  transl=torch.zeros(batch_size * seq_len, 3).to(device), return_vertices=False)
    root = output.joints[:, 11, :].reshape(batch_size, seq_len, 3)
    ksi_batch[..., :3] += root

    Rw2c_batch = G_batch[:, :3, :3]  # 16x3x3
    Tw2c_batch = G_batch[:, :3, -1]  # 16x3

    Wr2w_batch = ksi_batch[:, :, 3:]

    Rr2w_batch = torch.from_numpy(R.from_rotvec(Wr2w_batch.reshape(-1, 3).cpu()).as_matrix().astype(np.float32)).to(
        device).reshape(batch_size, seq_len, 3, 3)  # 16x100x3x3
    Tr2w_batch = ksi_batch[:, :, :3]  # 16x100x3

    Rr2c_batch = torch.einsum("bmn,bsnl->bsml", Rw2c_batch, Rr2w_batch)  # 16x100x3x3
    Tr2c_batch = torch.einsum("bmn,bsn->bsm", Rw2c_batch, Tr2w_batch) + Tw2c_batch[:, None, :]  # 16x100x3

    Wr2c_batch = torch.from_numpy(R.from_matrix(Rr2c_batch.reshape(-1, 3, 3).cpu()).as_rotvec().astype(np.float32)).to(
        device).reshape(batch_size, seq_len, 3)

    new_ksi_batch = torch.cat([Tr2c_batch, Wr2c_batch], 2)
    new_ksi_batch[..., :3] -= root
    return new_ksi_batch  # 16x100x6


# def test_transform_frame():
#     device = torch.device("cuda:0")
#     Rw2c = torch.from_numpy(R.from_rotvec([0,0,0]).as_matrix())
#     Gw2c = torch.cat([Rw2c, torch.Tensor([[3],[3],[3]])], 1)
#     Gw2c = torch.cat([Gw2c, torch.Tensor([[0,0,0,1]])], 0)
#     G_batch = torch.from_numpy(np.array([Gw2c.numpy()]*16)).to(device)
#     ksi_batch = torch.ones([16,100,6]).to(device)*0.5
#     ksi_new_batch = transform_frame(G_batch, ksi_batch, device)
#     print(ksi_new_batch[0,0,:])


# def transform_imu(Ri2c_batch, Rs2b_batch, reading_batch, device):# 16x3x3, 16x4x3x3, 16x100x40
#     batch_size = reading_batch.shape[0]
#     seq_len = reading_batch.shape[1]
#     n_imu = Rs2b_batch.shape[1]
#     reading_batch = reading_batch.reshape(batch_size, seq_len, 4, 10)
#     reading_batch = reading_batch.permute([0, 2, 1, 3]) # 16x4x100x10
#     new_reading_batch = []
#
#     for i in range(batch_size):
#         Ri2c = Ri2c_batch[i] # 3x3
#         Rs2b = Rs2b_batch[i] # 4x3x3
#         reading = reading_batch[i] # 4x100x10
#         new_reading = []
#         for imu_id in range(n_imu):
#             quat = reading[imu_id, :, :4]
#             Rs2i_seq = torch.from_numpy(R.from_quat(quat.cpu()).as_matrix().astype(np.float32)).to(device) # 100x3x3
#
#             Rs2i_mat = Rs2i_seq.reshape(-1, 3) # 300x3
#             #print(R.from_quat(quat.cpu()).as_matrix().dtype)
#             Rb2i_mat = Rs2i_mat.mm(Rs2b[imu_id].transpose(0, 1)) # 300x3
#             Rb2i_mat = Rb2i_mat.reshape(-1, 3, 3).permute([1, 0, 2]).reshape(3, -1) # 3x300
#
#             Rb2c_mat = Ri2c.mm(Rb2i_mat) # 3x300
#             Rb2c_mat = Rb2c_mat.reshape(3, seq_len, 3).permute([1, 0, 2]).reshape(-1, 9) # 100x3x3
#             #print(Rb2c_mat[0].reshape(-1, 3,3)-Ri2c.mm(Rs2i_seq[0]).mm(Rs2b[imu_id].transpose(0, 1)))
#             #print(Rb2c_mat[0].reshape(3,3)-Ri2c.mm(Rs2i_seq[0]).mm(Rs2b[imu_id].transpose(0, 1)))
#             #print(Ri2c.mm(Rs2i_seq[0]).mm(Rs2b[imu_id].transpose(0, 1)))
#             #print(Rb2i_mat[0].reshape(3,3))
#
#             omega = reading[imu_id,:,4:7]
#             omega = Ri2c.mm(omega.transpose(0, 1)).transpose(0, 1) # 100x3
#
#             accel = reading[imu_id,:,7:10] # 100x3
#             accel = Ri2c.mm(accel.transpose(0, 1)).transpose(0, 1) # 100x3
#
#             new_reading.append(torch.cat([Rb2c_mat, omega, accel], 1)[None,:,:])
#         new_reading = torch.cat(new_reading, 0)
#         new_reading_batch.append(new_reading[None,:,:,:])
#     new_reading_batch = torch.cat(new_reading_batch, 0)
#             #reading_new = np.concatenate([orient, accel, omega], 1)
#             #reading_new_list.append(reading_new)
#     return new_reading_batch.permute([0, 2, 1, 3]).reshape(batch_size, seq_len, -1)


def transform_imu_v2(Ri2c_batch, Rs2b_batch, reading_batch):  # 16x3x3, 16x4x3x3, 16x100x40
    device = reading_batch.device
    batch_size = reading_batch.shape[0]
    seq_len = reading_batch.shape[1]
    n_imu = Rs2b_batch.shape[1]
    reading_batch = reading_batch.reshape(batch_size, seq_len, 4, 7)
    reading_batch = reading_batch.permute([0, 2, 1, 3])  # 16x4x100x10

    quat_batch = reading_batch[..., :4]  # 16x4x100x4
    # omega_batch = reading_batch[..., 4:7]  # 16x4x100x3
    accel_batch = reading_batch[..., 4:]  # 16x4x100x3

    Rb2s_batch = Rs2b_batch.permute([0, 1, 3, 2])  # 16x4x3x3

    Rs2i_batch = torch.from_numpy(R.from_quat(quat_batch.reshape(-1, 4).cpu()).as_matrix().astype(np.float32)).to(
        device).reshape(batch_size, n_imu, seq_len, 3, 3)  # 16x4x100x3x3
    Rb2i_batch = torch.einsum("bismn,binl->bisml", Rs2i_batch, Rb2s_batch)  # 16x4x100x3x3

    Rb2c_batch = torch.einsum("bmn,bisnl->bisml", Ri2c_batch, Rb2i_batch).reshape(batch_size, n_imu, seq_len,
                                                                                  9)  # 16x4x100x9

    # omega_batch = torch.einsum("bmn,bisnl->bisml", Ri2c_batch, omega_batch[..., None])[..., 0]  # 16x4x100x3
    # accel_batch = torch.einsum("bmn,bisnl->bisml", Ri2c_batch, accel_batch[..., None])[..., 0] # 16x4x100x3

    # omega_batch = torch.einsum("bmn,bisn->bism", Ri2c_batch, omega_batch)  # 16x4x100x3
    accel_batch = torch.einsum("bmn,bisn->bism", Ri2c_batch, accel_batch)  # 16x4x100x3

    return torch.cat([Rb2c_batch, accel_batch], 3).permute([0, 2, 1, 3]).reshape(batch_size, seq_len, -1)


def get_b2r(pose_9d_batch):#16x100x23x9
    batch_size, seq_len, _, _ = pose_9d_batch.shape
    Rb2r_list = []
    motions = pose_9d_batch.reshape(-1, 24, 3, 3)# 1600x23x3x3
    for chain in KIN_CHAINS:
        rot_total = motions[:, chain[0], ...] # 1600x3x3
        for j in chain[1:-1]:
            rot = motions[:, j, ...] # 1600x3x3
            rot_total = torch.einsum("bmn,bnl->bml", rot, rot_total)
        Rb2r_list.append(rot_total.reshape(batch_size, seq_len, 9))
    return Rb2r_list

def get_b2c(pose_9d_batch):#16x100x23x9
    batch_size, seq_len, _, _ = pose_9d_batch.shape
    Rb2c_list = []
    motions = pose_9d_batch.reshape(-1, 24, 3, 3)# 1600x23x3x3
    for chain in KIN_CHAINS:
        rot_total = motions[:, chain[0], ...] # 1600x3x3
        for j in chain[1:]:
            rot = motions[:, j, ...] # 1600x3x3
            rot_total = torch.einsum("bmn,bnl->bml", rot, rot_total)
        Rb2c_list.append(rot_total.reshape(batch_size, seq_len, 9))
    return Rb2c_list


def batch_slerp(q1, q2, val=0.5, dim=-1):
    omega = torch.acos(
        ((q1 / q1.norm(dim=dim, keepdim=True)) * (q2 / q2.norm(dim=dim, keepdim=True))).sum(dim=dim, keepdim=True))
    #finfo = torch.finfo(q1.dtype)
    # if omega < finfo.eps:
    #     return q1
    so = torch.sin(omega)
    return torch.sin((1.0 - val) * omega) / so * q1 + torch.sin(val * omega) / so * q2


def batch_proj(joints3d_seq_batch, proj_batch):  # (8,100,21,3), (8,14, 3, 4)
    joints3d = torch.ones(joints3d_seq_batch.shape[0], joints3d_seq_batch.shape[1], 1,
                           joints3d_seq_batch.shape[2], dtype=torch.float32,
                           device=joints3d_seq_batch.device)  # (8,100,1,21)
    joints3d = torch.cat([joints3d_seq_batch.permute([0, 1, 3, 2]), joints3d], 2)  # (8,100,4,21)

    joints2d = torch.einsum("bvmn,bsnl->bvsml", proj_batch, joints3d)  # (8,100,3,21)

    joints2d = joints2d / joints2d[..., 2:3, :]
    return joints2d.permute([0, 1, 2, 4, 3])




















