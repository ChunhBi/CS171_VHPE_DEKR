import torch

def mse_loss(batch_pred, batch, batch_weight=None):
    residual = batch_pred - batch #(...,3, 3) (...,3, 3)
    loss = residual.pow(2).sum(dim=-1)
    if batch_weight is not None:
        loss = loss * batch_weight
    loss = loss.mean()
    return loss

def euclidean_loss(batch_pred, batch, batch_weight=None):
    residual = batch_pred - batch #(...,3, 3) (...,3, 3)
    loss = residual.norm(dim=-1)
    if batch_weight is not None:
        loss = loss * batch_weight
    loss = loss.mean()
    return loss


def hybrid_loss(output, train_data):
    lambda_vel = 10
    lambda_acc = 10.
    lambda_j3d = 10.
    lambda_j2d = 0.01
    lambda_regr = 1.
    lambda_r2c_net = 0.1
    lambda_r2c_final = 0.1
    lambda_b2c = 0.1

    vel_loss = mse(output["joints3d_vel"], torch.zeros_like(output["joints3d_vel"]))
    acc_loss = mse(output["joints3d_acc"], torch.zeros_like(output["joints3d_acc"]))
    j3d_loss = euclidean(output["joints3d"], train_data["joints3d"][:,1:,...])
    regr_loss = mse(output["fullpose_9d"], train_data["fullpose_9d"][:,1:,...])


    joints2d_allview = train_data["joints2d_allview"][:,:,1:,:,:]
    j2d_loss = euclidean(output["joints2d_allview"][...,:2], joints2d_allview[...,:2], joints2d_allview[...,2])

    r2c_net_loss =  mse(output["r2c_9d_net"], train_data["fullpose_9d"][:,1:,:1, :]) #+ loss_fn(r2c_final_pred, dof[:,1:,3:6])
    r2c_final_loss =  mse(output["r2c_9d_final"], train_data["fullpose_9d"][:,1:,:1, :])
    b2c1_loss =  mse(output["b2c_list"][0].view(batch_size, -1, 9), train_data["b2c_list"][0].view(batch_size, -1, 9))
    b2c2_loss =  mse(output["b2c_list"][1].view(batch_size, -1, 9), train_data["b2c_list"][1].view(batch_size, -1, 9))
    b2c3_loss =  mse(output["b2c_list"][2].view(batch_size, -1, 9), train_data["b2c_list"][2].view(batch_size, -1, 9))
    b2c4_loss =  mse(output["b2c_list"][3].view(batch_size, -1, 9), train_data["b2c_list"][3].view(batch_size, -1, 9))
    #r2c_list_loss =  mse(output["r2c_9d_net"], train_data["fullpose_9d"][:,1:,:1, :])
    b2c_loss = b2c1_loss + b2c2_loss + b2c3_loss + b2c4_loss

    loss =  lambda_regr * regr_loss\
            + lambda_r2c_net*r2c_net_loss\
            + lambda_j3d * j3d_loss + lambda_j2d * j2d_loss\
            + lambda_b2c * b2c_loss \
            + lambda_acc * acc_loss  + lambda_vel * vel_loss #+  r2c_loss#limb_loss +torso_loss +j3d_loss + regr_loss
    
    

    