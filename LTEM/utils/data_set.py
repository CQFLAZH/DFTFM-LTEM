from torch.utils.data import Dataset
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label, entity = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label']), np.int32(ele['entity'])

        label = self.get_label(label)
        entity = self.get_entity(entity)
        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)
            entity = (1.0 - self.label_smooth) * entity + (1.0 / self.num_ent)
        return triple, label, entity

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

    def get_entity(self, entity):

        y = np.zeros([self.num_ent], dtype=np.float32)
        y[entity] = 1
        return torch.tensor(y, dtype=torch.float32)



class TestDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TestDataset, self).__init__()
        self.triplets = triplets
        self.num_ent = num_ent

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label= torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        label = self.get_label(label)
        return triple, label

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)



# class TestDataset(Dataset):
#     def __init__(self, triplets, num_ent, params):
#         super(TestDataset, self).__init__()
#         self.triplets = triplets
#         self.num_ent = num_ent
#
#     def __len__(self):
#         return len(self.triplets)
#
#     def __getitem__(self, item):
#         ele = self.triplets[item]
#         triple, label, entity = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label']), np.int32(ele['entity'])
#         label = self.get_label(label)
#         entity = self.get_entity(entity)
#         return triple, label, entity
#
#     def get_label(self, label):
#         """
#         get label corresponding to a (sub, rel) pair
#         :param label: a list containing indices of objects corresponding to a (sub, rel) pair
#         :return: a tensor of shape [nun_ent]
#         """
#         y = np.zeros([self.num_ent], dtype=np.float32)
#         y[label] = 1
#         return torch.tensor(y, dtype=torch.float32)
#
#     def get_entity(self, entity):
#
#         y = np.zeros([self.num_ent], dtype=np.float32)
#         y[entity] = 1
#         return torch.tensor(y, dtype=torch.float32)
