import torch
import h5py
from Bio import SeqUtils
AA_TO_INDEX = {aa: i for i, aa in enumerate(SeqUtils.IUPACData.protein_letters_3to1.values())}

def decode(x):
    INDEX_TO_AA  = {i: aa for aa, i in AA_TO_INDEX.items()} 
    seq = ""
    for i in x.detach().tolist():
        seq += INDEX_TO_AA[i]
                                    
    return seq

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
    def __init__(self, hdf5_file):
        self.file = h5py.File(hdf5_file, 'r')
        self.len = self.file['len'][()]
         
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        seq = self.file['seqs'][idx].decode("utf-8")
        h = torch.tensor([AA_TO_INDEX[i] for i in seq])
        
        return Data(h)
