import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random

class MoleculeProteinDataset(Dataset):
    def __init__(self, allele_features, substrate_features, df, label_name = 'Revised_mean'):
        """
        Args:
            allele_features: Dictionary containing allele features
                {'cyp2d6*1': {'embedding': np.array}, ...}
            substrate_features: Dictionary containing substrate features
                {'sparteine': {'embedding': np.array}, ...}
            df: DataFrame containing pairs of allele and substrate
        """
        self.allele_features = allele_features
        self.substrate_features = substrate_features
        
        # DataFrame에서 유효한 pair만 필터링
        valid_pairs = []
        for _, row in df.iterrows():
            allele = row['Allele']
            substrate = row['Substrate']#.replace(' ', '')
            if allele in allele_features and substrate in substrate_features:
                valid_pairs.append({
                    'allele': allele,
                    'substrate': substrate,
                    'value': row[label_name]  # 필요한 경우 target value도 저장
                })
        self.valid_pairs = valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        allele = pair['allele']
        substrate = pair['substrate']
        
        return {
            'allele': torch.FloatTensor(self.allele_features[allele]),
            'substrate': torch.FloatTensor(self.substrate_features[substrate]),
            'allele_name': allele,
            'substrate_name': substrate,
            'value': pair['value']
        }


def MoleculeProteinCollate(batch):
    """
    Padds batch of variable length
    """
    alleles = [item['allele'] for item in batch]
    substrates = [item['substrate'] for item in batch]
    allele_names = [item['allele_name'] for item in batch]
    substrate_names = [item['substrate_name'] for item in batch]
    values = [item['value'] for item in batch]
    
    # Pad sequences
    padded_alleles = pad_sequence(alleles, batch_first=True)
    padded_substrates = pad_sequence(substrates, batch_first=True)
    
    # Create attention masks
    allele_masks = torch.zeros(len(alleles), padded_alleles.shape[1], dtype=torch.bool)
    substrate_masks = torch.zeros(len(substrates), padded_substrates.shape[1], dtype=torch.bool)
    
    for i, (allele, substrate) in enumerate(zip(alleles, substrates)):
        allele_masks[i, :len(allele)] = 1
        substrate_masks[i, :len(substrate)] = 1
    
    return {
        'alleles': padded_alleles,
        'substrates': padded_substrates,
        'allele_masks': allele_masks,
        'substrate_masks': substrate_masks,
        'allele_names': allele_names,
        'substrate_names': substrate_names,
        'values': torch.FloatTensor(values)
    }

