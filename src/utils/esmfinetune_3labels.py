import torch




class VarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings1, encodings2, labels):
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.labels = labels

    def __getitem__(self, idx):
        item1 = {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()}
        item2 = {key: torch.tensor(val[idx]) for key, val in self.encodings2.items()}
        label = torch.tensor(self.labels[idx])
        return (item1, item2, label)

    def __len__(self):
        return len(self.labels)
    
    
                                                                                 
class VarCollator(object):
    def __init__(self):
        """
        Collator for data
        """
        None
        

    def __call__(self, data):

        seq1 = [tp[0] for tp in data]
        seq2 = [tp[1] for tp in data]
        labels = [tp[2] for tp in data]
        
        batch = {}
        for k, v in seq1[0].items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch["seq1_"+k] = torch.stack([f[k] for f in seq1])
               
                
        for k, v in seq2[0].items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch["seq2_"+k] = torch.stack([f[k] for f in seq2])
      

        batch['labels'] = torch.FloatTensor(labels)
        
        return batch
import torch
class MultiTaskVarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings1, encodings2, labels_task1, labels_task2, labels_task3):
        """
        Dataset for multi-task learning with two tasks
        
        Args:
            encodings1: Dictionary of encodings for sequence 1
            encodings2: Dictionary of encodings for sequence 2
            labels_task1: List of labels for task 1
            labels_task2: List of labels for task 2
        """
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.labels_task1 = labels_task1
        self.labels_task2 = labels_task2
        self.labels_task3 = labels_task3
        
        # Sanity check to ensure all data has the same length
        assert len(labels_task1) == len(labels_task2), "Labels for both tasks must have the same length"
        
    def __getitem__(self, idx):
        item1 = {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()}
        item2 = {key: torch.tensor(val[idx]) for key, val in self.encodings2.items()}
        
        # Create a 2D label tensor with both task labels
        label = torch.tensor([self.labels_task1[idx], self.labels_task2[idx], self.labels_task3[idx]], dtype=torch.float)
        
        return (item1, item2, label)
    
    def __len__(self):
        return len(self.labels_task1)
    
class MultiTaskVarCollator(object):
    def __init__(self):
        """
        Collator for multi-task data
        """
        pass
        
    def __call__(self, data):
        seq1 = [tp[0] for tp in data]
        seq2 = [tp[1] for tp in data]
        labels = [tp[2] for tp in data]  # Each label is now a tensor of shape [2]
        
        batch = {}
        for k, v in seq1[0].items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch["seq1_"+k] = torch.stack([f[k] for f in seq1])
               
                
        for k, v in seq2[0].items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch["seq2_"+k] = torch.stack([f[k] for f in seq2])
      
        # Stack the labels into a batch_size x 2 tensor
        batch['labels'] = torch.stack(labels)
        
        return batch    

# class MultiTaskVarDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings1, encodings2, labels_task1, labels_task2):
#         """
#         Dataset for multi-task learning with two tasks
        
#         Args:
#             encodings1: Dictionary of encodings for sequence 1
#             encodings2: Dictionary of encodings for sequence 2
#             labels_task1: List of labels for task 1
#             labels_task2: List of labels for task 2
#         """
#         self.encodings1 = encodings1
#         self.encodings2 = encodings2
#         self.labels_task1 = labels_task1
#         self.labels_task2 = labels_task2
        
#         # Sanity check to ensure all data has the same length
#         assert len(labels_task1) == len(labels_task2), "Labels for both tasks must have the same length"
        
#     def __getitem__(self, idx):
#         item1 = {key: torch.tensor(val[idx]) for key, val in self.encodings1.items()}
#         item2 = {key: torch.tensor(val[idx]) for key, val in self.encodings2.items()}
        
#         # Create a 2D label tensor with both task labels
#         label = torch.tensor([self.labels_task1[idx], self.labels_task2[idx]], dtype=torch.float)
        
#         return (item1, item2, label)
    
#     def __len__(self):
#         return len(self.labels_task1)
    
# class MultiTaskVarCollator(object):
#     def __init__(self):
#         """
#         Collator for multi-task data
#         """
#         pass
        
#     def __call__(self, data):
#         seq1 = [tp[0] for tp in data]
#         seq2 = [tp[1] for tp in data]
#         labels = [tp[2] for tp in data]  # Each label is now a tensor of shape [2]
        
#         batch = {}
#         for k, v in seq1[0].items():
#             if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
#                 if isinstance(v, torch.Tensor):
#                     batch["seq1_"+k] = torch.stack([f[k] for f in seq1])
               
                
#         for k, v in seq2[0].items():
#             if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
#                 if isinstance(v, torch.Tensor):
#                     batch["seq2_"+k] = torch.stack([f[k] for f in seq2])
      
#         # Stack the labels into a batch_size x 2 tensor
#         batch['labels'] = torch.stack(labels)
        
#         return batch    

        
class MultiTaskVarOnlyCollator(object):
    def __init__(self):
        """
        Collator for multi-task data
        """
        pass
        
    def __call__(self, data):
        seq = [tp[0] for tp in data]
        labels = [tp[1] for tp in data]  # Each label is now a tensor of shape [2]
        
        batch = {}
        for k, v in seq[0].items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch["seq_"+k] = torch.stack([f[k] for f in seq])
               
      
        # Stack the labels into a batch_size x 2 tensor
        batch['labels'] = torch.stack(labels)
        
        return batch


class MultiTaskVarOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels_task1, labels_task2, labels_task3, labels_task4):
        """
        Dataset for multi-task learning with two tasks
        
        Args:
            encodings1: Dictionary of encodings for sequence 1
            encodings2: Dictionary of encodings for sequence 2
            labels_task1: List of labels for task 1
            labels_task2: List of labels for task 2
        """
        self.encodings = encodings
        self.labels_task1 = labels_task1
        self.labels_task2 = labels_task2
        self.labels_task3 = labels_task3
        self.labels_task4 = labels_task4
        
        # Sanity check to ensure all data has the same length
        assert len(labels_task1) == len(labels_task2), "Labels for both tasks must have the same length"
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        
        # Create a 2D label tensor with both task labels
        label = torch.tensor([self.labels_task1[idx], self.labels_task2[idx], self.labels_task3[idx], self.labels_task4[idx]], dtype=torch.float)
        
        return (item, label)
    
    def __len__(self):
        return len(self.labels_task1)
    
    


from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def eval_reg_metrics_nosig(predictions, labels):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    #sigmoid = torch.nn.Sigmoid()
    y_pred = predictions.flatten()
    # next, use threshold to turn them into integer predictions
    # finally, compute metrics
    y_true = labels.flatten()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    rsq = r2_score(y_true, y_pred)
    # return as dictionary
    metrics = {'rmse': rmse,
               'r2': rsq}
    return metrics

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def eval_multitask_reg_metrics(predictions, labels):
    """
    Compute regression metrics for multiple tasks
    
    Args:
        predictions: tensor of shape (batch_size, num_tasks)
        labels: tensor of shape (batch_size, num_tasks)
        
    Returns:
        Dictionary with metrics for each task and averages
    """
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    # Convert tensors to numpy if they aren't already
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().detach().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().detach().numpy()
    
    num_tasks = predictions.shape[1]
    metrics = {}
    
    # Calculate metrics for each task
    for task_idx in range(num_tasks):
        y_pred = predictions[:, task_idx]
        if task_idx == 0:
            y_pred = sigmoid(y_pred)
        y_true = labels[:, task_idx]
        
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        
        metrics[f'rmse_task{task_idx+1}'] = rmse
        metrics[f'r2_task{task_idx+1}'] = r2
    
    # Calculate average R² score across all tasks
    metrics['r2_avg'] = np.mean([metrics[f'r2_task{task_idx+1}'] for task_idx in range(num_tasks)])
    
    return metrics