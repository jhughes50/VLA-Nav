"""
    CIS 6200 -- Deep Learning Final Project
    class for data interpolation
    April 2024
"""
import numpy as np

class Interpolator:

    def __init__(self, output_dim):
        assert output_dim % 2 == 0, "[Interpolator] output dimension must be even"
        assert output_dim % 3 == 0, "[Interpolator] output dimension must be a multiple of 3"
        self.out_dim_ = int(output_dim / 3)



    def interpolate_batch(self, paths):        
        out_poses = np.empty(( self.out_dim_, 3))
        num_poses = paths.shape[1]
        
        total_out = np.empty((paths.shape[0], self.out_dim_*3))

        for j in range(paths.shape[0]):
            poses = paths[j]
            if num_poses == 0:
                return None
            elif num_poses == 1:
                out_poses = np.ones((self.out_dim_, 3))
                out_poses = poses * out_poses
            elif num_poses == 2:
                out_poses = self.subinterpolate(poses[0], poses[1], self.out_dim_)
            elif num_poses == 3:
                split = self.out_dim_//2
                out_poses[0:split] = self.subinterpolate(poses[0], poses[1], split)
                out_poses[split:-1] = self.subinterpolate(poses[1], poses[2], split)[1:,:]
            else:
                remainder = self.out_dim_ % (num_poses-1)
                split = self.out_dim_ // (num_poses-1)
                
                for i in range(num_poses-1):
                    if remainder != 0 and i+1 == num_poses-1:
                        out_poses[split*i:,:] = self.subinterpolate(poses[i], poses[i+1],split+remainder)
                    else:
                        out_poses[split*i:split*(i+1),:] = self.subinterpolate(poses[i],poses[i+1],split)
            total_out[j] = out_poses.flatten()

        return total_out


    def interpolate(self, poses):        
        out_poses = np.empty((self.out_dim_, 3))
        num_poses = len(poses)
        
        if num_poses == 0:
            return None
        elif num_poses == 1:
            out_poses = np.ones((self.out_dim_, 3))
            out_poses = poses * out_poses
        elif num_poses == 2:
            out_poses = self.subinterpolate(poses[0], poses[1], self.out_dim_)
        elif num_poses == 3:
            split = self.out_dim_//2
            out_poses[0:split] = self.subinterpolate(poses[0], poses[1], split)
            out_poses[split:-1] = self.subinterpolate(poses[1], poses[2], split)[1:,:]
        else:
            remainder = self.out_dim_ % (num_poses-1)
            split = self.out_dim_ // (num_poses-1)
            
            for i in range(num_poses-1):
                if remainder != 0 and i+1 == num_poses-1:
                    out_poses[split*i:,:] = self.subinterpolate(poses[i], poses[i+1],split+remainder)
                else:
                    out_poses[split*i:split*(i+1),:] = self.subinterpolate(poses[i],poses[i+1],split)

        return out_poses.flatten()


    def subinterpolate(self, start, end, dim):
        return np.linspace(start, end, dim)
