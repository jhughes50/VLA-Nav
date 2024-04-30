"""
    CIS 6200 Final Project
    Extract poses from paths for autoencoder
    April 2024
"""

import numpy as np
import gzip
import json
#from utils.matcher import Matcher

class Matcher:

    def __init__(self):
        self.uids_ = None


    def match(self, ids):
        ids, uids = np.unique(ids, return_index=True)
        self.uids_ = np.sort(uids)


    def poses_from_match(self, poses):
        return poses[self.uids_]


class Interpolator:

    def __init__(self, output_dim):
        assert output_dim % 2 == 0, "[Interpolator] output dimension must be even"
        assert output_dim % 3 == 0, "[Interpolator] output dimension must be a multiple of 3"
        self.out_dim_ = int(output_dim / 3)


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


class PoseExtractor:

    def __init__(self, path, out_dim=96):
        self.pose_ = None
        self.train_guide_ = list()
        self.generic_path_ = path

        self.matcher_ = Matcher()
        self.interpolator_ = Interpolator(out_dim)

        #self.load(path)

    def interpolate(self, path):
        return self.interpolator_.interpolate(path)


    def load(self):
        path = self.generic_path_ + "rxr-data/rxr_train_guide.jsonl.gz"
        with gzip.open(path, 'r') as f:
            print("[PoseExtractor] found file %s, loading..." %path)
            self.train_guide_ = [json.loads(line) for line in f]
            print("[PoseExtractor] guide loaded")        


    def get_path_poses(self, pose_path):
        try:
            pose_trace = np.load(pose_path)
            poses = pose_trace["extrinsic_matrix"][:,:-1,-1]
        except FileNotFoundError:
            print("file not found at %s" %pose_path)
            return np.array([])
        self.matcher_.match(pose_trace["pano"])
        unique_poses = self.matcher_.poses_from_match(poses)
        return unique_poses

    def get_path(self, subguide):
        inst_id = subguide['instruction_id']
        pose_path = self.generic_path_+ \
            "rxr-data/pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(inst_id)

        return self.get_path_poses(pose_path)

    def path_from_guide(self, idx):
        # here we can get the path of UIDs from the train guide
        # but then where to we get the actual poses of the points on the path.
        assert len(self.train_guide_) > 0, "[PoseExtractor] Training Guide is not loaded."

        guide = self.train_guide_[idx]
        
        instruction_id = guide["instruction_id"]
        pose_path = self.generic_path_+ \
            "rxr-data/pose_traces/rxr_train/{:06}_guide_pose_trace.npz".format(instruction_id)
            
        unique_poses = self.get_path_poses(pose_path)
        return unique_poses

