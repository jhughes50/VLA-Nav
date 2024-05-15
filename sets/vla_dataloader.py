"""
    CIS 6200 -- Deep Learning Final Project
    dataloader for vision, language and path
    April 2024
"""
import torch

class VLADataLoader:

    def __init__(self, dataset, batch_size=32, path_shape=(32,3), img_shape = (240, 320, 3)):

        self.dataset_ = dataset
        self.batch_size_ = batch_size
        self.index_ = 0
        self.path_shape_ = path_shape
        self.img_shape_ = img_shape

    def __len__(self):
        return len(self.dataset_.be.guide)#// self.batch_size_

    def __getitem__(self, idx):
        return self.batchify(idx)

    def batchify(self, idx):
        # loop and create combinations
        fill = 0
        
        img_tensor = torch.zeros(tuple([self.batch_size_]) + self.img_shape_) 
        path_tensor = torch.zeros(tuple([self.batch_size_]) + self.path_shape_)
        text_list = list()
        map_list = list()

        count = 0

        while fill < self.batch_size_: 
            data = self.dataset_[self.index_]
            # if the text is not english skip it.
            if data.text == None:
                self.index_ += 1
                continue
            
            self.index_ += 1

            imgs = data.image
            path = data.path
            text = data.text

            for i, img in enumerate(imgs):
                img_tensor[fill] = torch.tensor(img)[:,:,:-1]
                full_path = self.dataset_.be.interpolate(path[i:])
                path_tensor[fill] = torch.tensor(full_path.reshape(self.path_shape_))
                text_list.append(text)
                           
                map_list.append(count)

                fill += 1
                if fill == self.batch_size_:
                    break
            count += 1
            label_tensor = torch.tensor(map_list)

        return text_list, img_tensor, path_tensor, label_tensor
