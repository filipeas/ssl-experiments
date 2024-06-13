import torch
import numpy as np
from transforms3d.axangles import axangle2mat
from scipy.interpolate import CubicSpline 

def da_permutation(X, nPerm=5, minSegLength=7):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile is True:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(
            np.random.randint(
                minSegLength, X.shape[0] - minSegLength, nPerm - 1
            )
        )
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]] : segs[idx[ii] + 1], :]
        X_new[pp : pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new

def da_scaling(X, sigma=0.3, min_scale_sigma=0.05):
    scaling_factor = np.random.normal(
        loc=1.0, scale=sigma, size=(1, X.shape[1])
    )  # shape=(1,3)

    while np.any(np.abs(scaling_factor - 1) < min_scale_sigma):
        scaling_factor = np.random.normal(
            loc=1.0, scale=sigma, size=(1, X.shape[1])
        )
    my_noise = np.matmul(np.ones((X.shape[0], 1)), scaling_factor)
    X = X * my_noise
    return X

def generate_random_curves(X, sigma=0.2, knot=4):
    xx = (
        np.ones((X.shape[1], 1))
        * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
    ).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()

def distort_timesteps(X, sigma=0.2):
    tt = generate_random_curves(
        X, sigma
    )  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [
        (X.shape[0] - 1) / tt_cum[-1, 0],
        (X.shape[0] - 1) / tt_cum[-1, 1],
        (X.shape[0] - 1) / tt_cum[-1, 2],
    ]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum

def da_time_warp(X, sigma=0.4):
    tt_new = distort_timesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new


class Rotation: 
    def __call__(self, x: torch.Tensor):
        axes = np.identity(3)
        sampleAcc = x[0:3].numpy()
        sampleGyr = x[3:6].numpy()

        sampleAcc = np.swapaxes(sampleAcc, 0, 1)
        sampleGyr = np.swapaxes(sampleGyr, 0, 1)

        for i in range(3):
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            sampleAcc = np.matmul(sampleAcc, axangle2mat(axes[i], angle))
            sampleGyr = np.matmul(sampleGyr, axangle2mat(axes[i], angle))

        sampleAcc = np.swapaxes(sampleAcc, 0, 1)
        sampleGyr = np.swapaxes(sampleGyr, 0, 1)
        return torch.Tensor(np.concatenate((sampleAcc, sampleGyr)))
    def getName(self):
        return "Rotation"

rotation = Rotation()

class Flip: 
    def __call__(self, x: torch.Tensor):
        sampleAcc = x[0:3].numpy()
        sampleGyr = x[3:6].numpy()

        sampleAcc = np.flip(sampleAcc, 1)
        sampleGyr = np.flip(sampleGyr, 1)
        return torch.Tensor(np.concatenate((sampleAcc, sampleGyr)))
    def getName(self):
        return "Flip"

flip = Flip()

class Noise_addition: 
    def __call__(self, x: torch.Tensor):
        new_x = x.clone()
        for i in range(x.shape[0]):
            new_x[i].add_(torch.tensor([np.random.normal(loc=0, scale=0.02) for _ in range(new_x[i].numel())], dtype=new_x[i].dtype))
        return new_x
    def getName(self):
        return "Noise_addition"
    
noise_addition = Noise_addition()

class Permutation:
    def __call__(self, x: torch.Tensor): 
        sampleAcc = x[0:3].numpy()
        sampleGyr = x[3:6].numpy()

        sampleAcc = np.swapaxes(sampleAcc, 0, 1)
        sampleGyr = np.swapaxes(sampleGyr, 0, 1)

        sampleAcc = da_permutation(X=sampleAcc)
        sampleGyr = da_permutation(X=sampleGyr)

        sampleAcc = np.swapaxes(sampleAcc, 0, 1)
        sampleGyr = np.swapaxes(sampleGyr, 0, 1)
        
        return torch.Tensor(np.concatenate((sampleAcc, sampleGyr)))
    
    def getName(self):
        return "Permutation"

permutation = Permutation()

class Scaling: 
    def __call__(self, x: torch.Tensor):
        sampleAcc = x[0:3].numpy()
        sampleGyr = x[3:6].numpy()

        sampleAcc = da_scaling(X=sampleAcc)
        sampleGyr = da_scaling(X=sampleGyr)
        
        return torch.Tensor(np.concatenate((sampleAcc, sampleGyr)))
    def getName(self):
        return "Scaling"

scaling = Scaling()

class TimeWarp: 
    def __call__(self, x: torch.Tensor):
        sampleAcc = x[0:3].numpy()
        sampleGyr = x[3:6].numpy()

        sampleAcc = np.swapaxes(sampleAcc, 0, 1)
        sampleGyr = np.swapaxes(sampleGyr, 0, 1)

        sampleAcc = da_time_warp(X=sampleAcc)
        sampleGyr = da_time_warp(X=sampleGyr)

        sampleAcc = np.swapaxes(sampleAcc, 0, 1)
        sampleGyr = np.swapaxes(sampleGyr, 0, 1)
        
        return torch.Tensor(np.concatenate((sampleAcc, sampleGyr)))
        
    def getName(self):
        return "Time Warp"
    
time_warp = TimeWarp()

class Negation: 
    def __call__(self, x: torch.Tensor):
        new_x = x.clone()
        new_x.mul_(-1)
        return new_x
    def getName(self):
        return "Negation"

negation = Negation()

