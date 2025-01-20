import torch
from torch.utils.data import Dataset,DataLoader




class PAVC_Dataset_Train(Dataset):
    
    def __init__(self, smiles_list,  graphs, ecfps, mds, label_list):
        super().__init__()
        # molecular fingerprints
        self.fps = ecfps
        #molecular descriptor
        self.mds = mds
        # molecular graphs
        self.graphs = graphs
        self.smiless = smiles_list # drug smiles

        self.labels = torch.tensor(label_list, dtype=torch.float32)

    def __len__(self):
        return len(self.smiless)
    
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx], self.labels[idx]



class PAVC_Dataset_Infer(Dataset):

    def __init__(self, smiles_list,  graphs, ecfps, mds):
        super().__init__()
        # molecular fingerprints
        self.fps = ecfps
        #molecular descriptor
        self.mds = mds
        # molecular graphs
        self.graphs = graphs
        self.smiless = smiles_list # drug smiles

    def __len__(self):
        return len(self.smiless)
    
    def __getitem__(self, idx):
        return self.smiless[idx], self.graphs[idx], self.fps[idx], self.mds[idx]
    






    
