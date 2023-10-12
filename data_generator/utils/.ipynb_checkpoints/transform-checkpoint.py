from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
    
def transform_frame(G_batch, ksi_batch, device):# 16x4x4 16x100x6
    batch_size = ksi_batch.shape[0]
    seq_len = ksi_batch.shape[1]
    ksi_new_list = []
    for i in range(batch_size):
        G = G_batch[i]
        rot = torch.from_numpy(R.from_rotvec(ksi_batch[i, :, 3:].cpu()).as_matrix().astype(np.float32)).to(device)
        trans = ksi_batch[i, :, :3][:,:,None]
        G_origin = torch.cat([rot, trans], 2)
        homo = torch.Tensor([[[0,0,0,1]]]*seq_len).to(device)
        G_origin = torch.cat([G_origin, homo], 1)

        G_new = G.mm(G_origin.permute([1,0,2]).reshape(4, -1))
        G_new = G_new.reshape(4, seq_len, 4)
        G_new = G_new.permute([1,0,2])
        
        #print(G_new[0] - G.mm(G_origin[0]))
        
        rotvec = torch.from_numpy(R.from_matrix(G_new[:, :3, :3].cpu()).as_rotvec().astype(np.float32)).to(device)
        trans = G_new[:, :3, -1]
        ksi_new = torch.cat([trans, rotvec], 1)
        ksi_new_list.append(ksi_new[None,:,:])
    return torch.cat(ksi_new_list, 0)



def test_transform_frame():
    device = torch.device("cuda:0")
    Rw2c = torch.from_numpy(R.from_rotvec([0,0,0]).as_matrix())
    Gw2c = torch.cat([Rw2c, torch.Tensor([[3],[3],[3]])], 1)
    Gw2c = torch.cat([Gw2c, torch.Tensor([[0,0,0,1]])], 0)
    G_batch = torch.from_numpy(np.array([Gw2c.numpy()]*16)).to(device)
    ksi_batch = torch.ones([16,100,6]).to(device)*0.5
    ksi_new_batch = transform_frame(G_batch, ksi_batch, device)
    print(ksi_new_batch[0,0,:])



def transform_imu(Ri2c_batch, Rs2b_batch, reading_batch, device):# 16x3x3, 16x4x3x3, 16x100x40
    batch_size = reading_batch.shape[0]
    seq_len = reading_batch.shape[1]
    n_imu = Rs2b_batch.shape[1]
    reading_batch = reading_batch.reshape(batch_size, seq_len, 4, 10)
    reading_batch = reading_batch.permute([0, 2, 1, 3]) # 16x4x100x10
    new_reading_batch = []
    
    for i in range(batch_size):
        Ri2c = Ri2c_batch[i] # 3x3
        Rs2b = Rs2b_batch[i] # 4x3x3
        reading = reading_batch[i] # 4x100x10
        new_reading = []
        for imu_id in range(n_imu):
            quat = reading[imu_id, :, :4]
            Rs2i_seq = torch.from_numpy(R.from_quat(quat.cpu()).as_matrix().astype(np.float32)).to(device) # 100x3x3

            Rs2i_mat = Rs2i_seq.reshape(-1, 3) # 300x3
            #print(R.from_quat(quat.cpu()).as_matrix().dtype)
            Rb2i_mat = Rs2i_mat.mm(Rs2b[imu_id].transpose(0, 1)) # 300x3
            Rb2i_mat = Rb2i_mat.reshape(-1, 3, 3).permute([1, 0, 2]).reshape(3, -1) # 3x300
            
            Rb2c_mat = Ri2c.mm(Rb2i_mat) # 3x300
            Rb2c_mat = Rb2c_mat.reshape(3, seq_len, 3).permute([1, 0, 2]).reshape(-1, 9) # 100x3x3
            #print(Rb2c_mat[0].reshape(-1, 3,3)-Ri2c.mm(Rs2i_seq[0]).mm(Rs2b[imu_id].transpose(0, 1)))
            #print(Rb2c_mat[0].reshape(3,3)-Ri2c.mm(Rs2i_seq[0]).mm(Rs2b[imu_id].transpose(0, 1)))
            #print(Ri2c.mm(Rs2i_seq[0]).mm(Rs2b[imu_id].transpose(0, 1)))
            #print(Rb2i_mat[0].reshape(3,3))
            
            omega = reading[imu_id,:,4:7]
            omega = Ri2c.mm(omega.transpose(0, 1)).transpose(0, 1) # 100x3

            accel = reading[imu_id,:,7:10] # 100x3
            accel = Ri2c.mm(accel.transpose(0, 1)).transpose(0, 1) # 100x3
            
            new_reading.append(torch.cat([Rb2c_mat, omega, accel], 1)[None,:,:])
        new_reading = torch.cat(new_reading, 0)
        new_reading_batch.append(new_reading[None,:,:,:])
    new_reading_batch = torch.cat(new_reading_batch, 0)
            #reading_new = np.concatenate([orient, accel, omega], 1)
            #reading_new_list.append(reading_new)
    return new_reading_batch.permute([0, 2, 1, 3]).reshape(batch_size, seq_len, -1)