"""
PyTorch-based SO(3) and SE(3) operations libs
"""
import torch
from DIVO.utils.torch_utils import b_div, b_mul

def skew(w):
    """_summary_
    w <-> [w] / R^3 <-> so(3)
    v <-> [v] / R^6 <-> se(3)
    """
    bs = w.size(0)
    if w.size() == (bs, 3, 3):
        W = torch.cat([
            -w[:, 1, 2:3],
            w[:, 0, 2:3],
            -w[:, 0, 1:2]
            ], dim=1)
    elif w.size() == (bs, 3):
        zero1 = torch.zeros(bs, 1, 1).to(w)
        w = w.view(bs, 3, 1, 1)
        W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
    elif w.size() == (bs, 4, 4):
        w_so3 = w[:, :3, :3]
        W = torch.cat([skew(w_so3), w[:, :3, 3]], dim=1)
    elif w.size() == (bs, 6):
        w_so3 = w[:, :3]
        w_temp = torch.cat(
            [skew(w_so3), w[:, 3:].view(-1, 3, 1)], dim=-1)
        zero1 = torch.zeros(bs, 1, 4).to(w)
        W = torch.cat([w_temp, zero1], dim=1)
    else:
        raise NotImplementedError
    return W

def exp_so3(x):
    """_summary_
    x -> exp([x])
    or
    [x] -> exp([x])
    """
    bs = x.size(0)
    if x.size() == (bs, 3, 3):
        W = x
        w = skew(W)
    elif x.size() == (bs, 3):
        w = x
        W = skew(w)
    else:
        raise NotImplementedError
    
    wnorm = torch.norm(w, dim=1)
    sw = torch.sin(wnorm)
    cw = torch.cos(wnorm)
    
    Id = torch.eye(3).to(x).view(1, 3, 3)
    
    eps = 1e-7
    idx = wnorm > eps 
    
    R = torch.zeros(bs, 3, 3).to(x)
    R[~idx] = Id + W[~idx] + 0.5*(W[~idx]@W[~idx])
    W_hat = b_div(W[idx], wnorm[idx])
    R[idx] = Id + b_mul(sw[idx], W_hat) + b_mul((1-cw[idx]), W_hat@W_hat)
    return R

def proj_minus_one_plus_one(x):
    eps = 1e-6
    x = torch.min(x, (1 - eps) * (torch.ones(x.shape).to(x)))
    x = torch.max(x, (-1 + eps) * (torch.ones(x.shape).to(x)))
    return x

def log_SO3(R):
    """_summary_
    R -> log(R)
    """
    eps = 1e-4
    trace = torch.sum(R[:, range(3), range(3)], dim=1)

    omega = R * torch.zeros(R.shape).to(R)

    theta = torch.acos(proj_minus_one_plus_one((trace - 1) / 2))

    temp = theta.unsqueeze(-1).unsqueeze(-1)

    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

    omega_vector_temp = torch.sqrt(omega_temp[:, range(3), range(3)] + torch.ones(3).to(R))
    A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
    B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
    C = omega_vector_temp[:, 0]
    omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= eps] = skew(omega_vector) * torch.pi
    return omega

def inv_SO3(R):
    """_summary_
    R -> R^-1
    """
    return R.permute(0, 2, 1)

def bracket_so3(x):
    # Extract the upper triangle elements that represent the skew-symmetric values
    return torch.stack([
        x[:, 2, 1],  # x component
        x[:, 0, 2],  # y component
        x[:, 1, 0]   # z component
    ], dim=1)

def exp_se3(x):
    """_summary_
    x = (w, v)
    ||w|| \neq 0 
    x -> exp([x])
    or
    [x] -> exp([x])
    """
    bs = x.size(0)
    if x.size() == (bs, 4, 4):
        W = x[:, :3, :3]
        w = skew(W)
        v = x[:, :3, 3]
    elif x.size() == (bs, 6):
        w = x[:, :3]
        v = x[:, 3:]
        W = skew(w)
    else:
        raise NotImplementedError
    
    eps = 1e-4
    wnorm = torch.norm(w, dim=1)
    
    idx = wnorm < eps
    
    v = v/torch.clip(wnorm.view(-1, 1), min=1e-5) # b_div(v, wnorm)

    wnorm = wnorm.view(-1, 1, 1)
    sw = torch.sin(wnorm)
    cw = torch.cos(wnorm)
    
    Id = torch.eye(3).to(x).view(1, 3, 3)
    W_squared = W@W
    W_hat = W/torch.clip(wnorm, min=1e-5)
    W_hat_squared = W_squared/torch.clip(wnorm**2, min=1e-5)
    W_idx = W[idx]
    wnorm_idx = wnorm[idx]
    W_squared_idx = W_squared[idx]
    
    ## compute R
    R = Id + sw*W_hat + (1 - cw)*W_hat_squared
    R[idx] = Id + W_idx + 0.5*W_squared_idx
    
    ## G(wnorm)
    G = Id*wnorm + (1 - cw)*W_hat + (wnorm - sw)*W_hat_squared
    G[idx] = Id*wnorm_idx
    p = G@v.view(-1, 3, 1)
    
    T = torch.cat([
        torch.cat([R, p], dim=-1),
        torch.tensor([[[0., 0., 0., 1.]]]).repeat(bs, 1, 1).to(x)
    ], dim=1)
    return T
    
def log_SE3(T):
    """_summary_
    T -> (w, v)
    """
    bs = T.size(0)
    R = T[:, 0:3, 0:3]
    p = T[:, 0:3, 3:]  
    
    W = log_SO3(R)  
    w = skew(W)  

    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(bs), range(bs)]]  
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  
    wnorm = torch.sqrt(wsqr)  
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  
    wnorm_inv = 1 / wnorm_unsqueezed  
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  

    P = torch.eye(3).to(W) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (
                wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
    v = torch.inverse(P) @ p  # n,3,1
    return torch.cat([torch.cat([W, v], dim=2), torch.zeros(bs, 1, 4).to(W)], dim=1)

def inv_SE3(T):
    """_summary_
    T -> T^-1
    """
    bs = len(T)
    R = T[:, :3, :3]
    p = T[:, :3, 3:]
    
    temp = torch.cat([R.permute(0, 2, 1), -R.permute(0, 2, 1)@p], dim=-1)
    invT = torch.cat([temp, torch.tensor(bs*[[[0., 0., 0., 1.]]]).to(T)], dim=1)
    return invT

def Adjoint(T):
    """_summary_
    T -> [Ad_T]
    """
    bs = len(T)
    R = T[:, :3, :3]
    p = T[:, :3, 3]
    zeros = torch.zeros(bs, 3, 3).to(T)
    AdT = torch.cat([
        torch.cat([R, zeros], dim=-1),
        torch.cat([skew(p)@R, R], dim=-1)    
    ], dim=1)
    return AdT

def adjoint(V):
    """_summary_
    V -> [ad_V]
    """
    bs = len(V)
    w = V[:, :3]
    v = V[:, 3:]
    zeros = torch.zeros(bs, 3, 3).to(V)
    adV = torch.cat([
        torch.cat([skew(w), zeros], dim=-1),
        torch.cat([skew(v), skew(w)], dim=-1)    
    ], dim=1)
    return adV

def approxmiate_pinv(J):
    """_summary_

    Args:
        J (torch.tensor): (bs, 6, dof)
    """
    bs, _, dof = J.size()
    temp = J@J.permute(0, 2, 1) + 1.0e-7*torch.eye(6).view(1, 6, 6).to(J)
    return J.permute(0, 2, 1)@torch.inverse(temp)