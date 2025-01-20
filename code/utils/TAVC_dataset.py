import torch
from torch.utils.data import Dataset,DataLoader


class TAVC_Dataset_Train(Dataset):
    def __init__(self, 
                 smiles_list, 
                 target_seq_list,
                 target_feature_dict,
                 label_list,  
                 graphs, 
                 fps, 
                 mds):
        super().__init__()

        self.smiless = smiles_list # drug smiles
        self.graphs = graphs # molecule graph
        self.fps = fps # molecule fingerprints
        self.mds = mds # molecule descriptors
        self.target_seq_list = target_seq_list # target seq 
        self.target_feature_dict = target_feature_dict # target feature
        self.labels = torch.tensor(label_list, dtype=torch.float32) # label
        

 
    def __len__(self):

        return len(self.smiless)
    
    def __getitem__(self, idx):

        smiles = self.smiless[idx]
        graphs = self.graphs[idx]
        mds = self.mds[idx]
        labels = self.labels[idx]
        fps = self.fps[idx]
        # 从 target_feature_id 中取出 ID 列表，并从 target_dict 中获取对应的蛋白质特征
        target_ids = self.target_seq_list[idx]  # 对应的蛋白质特征 ID
        # 如果 target_feature_ids 是列表，逐个从 target_dict 中获取特征
        if isinstance(target_ids, list):
            target_features = [self.target_feature_dict[id] for id in target_ids]
     
        else:
            # 如果 target_feature_ids 是单个ID
            target_features = self.target_feature_dict[target_ids]
        
        # target_features = torch.stack(target_features)
        return  smiles, graphs, fps, mds, target_features, labels
    


class TAVC_Dataset_infer(Dataset):
    def __init__(self, 
                 smiles_list, 
                 target_id_list,
                 target_feature_dict,
                 graphs, 
                 fps, 
                 mds):
        super().__init__()

        self.smiless = smiles_list # drug smiles
        self.graphs = graphs # molecule graph
        self.fps = fps # molecule fingerprints
        self.mds = mds # molecule descriptors
        self.target_id_list = target_id_list # target id
        self.target_feature_dict = target_feature_dict # target feature

    def __len__(self):

        return len(self.smiless)
    
    def __getitem__(self, idx):

        smiles = self.smiless[idx]
        graphs = self.graphs[idx]
        mds = self.mds[idx]
        fps = self.fps[idx]
        # 从 target_feature_id 中取出 ID 列表，并从 target_dict 中获取对应的蛋白质特征
        target_ids = self.target_id_list[idx]  # 对应的蛋白质特征 ID
        # 如果 target_feature_ids 是列表，逐个从 target_dict 中获取特征
        if isinstance(target_ids, list):
            target_features = [self.target_feature_dict[id] for id in target_ids]
     
        else:
            # 如果 target_feature_ids 是单个ID
            target_features = self.target_feature_dict[target_ids]
        
        # target_features = torch.stack(target_features)
        return  smiles, graphs, fps, mds, target_features, target_ids



