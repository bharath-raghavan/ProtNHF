import torch
import h5py
from Bio import SeqUtils

class Data:
    def __init__(self, h, batch=None, device='cpu'):
        self.h = h
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.int64, device=device)
        self.batch = batch
        self.device = device
    
    def to(self, device):
        h = self.h.to(device)
        batch = self.batch.to(device)
        return Data(h, batch, device)

class DataLoader(torch.utils.data.DataLoader):
   def __init__(
       self,
       dataset,
       batch_size: int = 1,
       shuffle: bool = False,
       **kwargs,
   ):
       super().__init__(
           dataset,
           batch_size,
           shuffle,
           collate_fn=self.collater,
           **kwargs,
       )       
               
   def collater(self, dataset):
        return Data(
               torch.cat([d.h for d in dataset]),
               batch=torch.cat([d.batch+i for i,d in enumerate(dataset)])
           )
                             
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.file = h5py.File('/lustre/orion/stf006/scratch/bharathrn/datasets/alphafold/human/dataset.h5', 'r')
        self.len = self.file['len'][()]
        self.AA_TO_INDEX = {aa: i for i, aa in enumerate(SeqUtils.IUPACData.protein_letters_3to1.values())}
        self.INDEX_TO_AA  = {i: aa for aa, i in self.AA_TO_INDEX.items()}
        self.mean = torch.tensor(self.file['mean'][:])
        self.std = torch.tensor(self.file['std'][:])
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        row = self.file[str(idx)]
        seq = row.attrs['seq']
        h = torch.tensor([self.AA_TO_INDEX[i] for i in seq]) #torch.tensor(row['h'][:]) #
        
        return Data(h)
    
    def decode(self, x):
        seq = ""
        for i in x.detach().tolist():
            seq += self.INDEX_TO_AA[i]
                                        
        return seq
