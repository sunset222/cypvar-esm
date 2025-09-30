import random
import os
import numpy as np
import torch
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    #print(f"Random seed set as {seed}")


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    first_cycle_epochs: int,
    min_lr: float = 0.0,
    warmup_start_factor: float = 0.0,
    cycle_mult: int = 1,
) -> SequentialLR:
    """
    PyTorch 내장 스케줄러들을 활용하여 warmup + cosine with restarts 스케줄러 생성
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: warmup 기간
        first_cycle_epochs: 첫 번째 cosine cycle의 길이
        min_lr: 최소 learning rate
        warmup_start_factor: warmup 시작 시점의 learning rate 비율 (0~1)
        cycle_mult: restart 후 cycle 길이 증가 비율
    """
    # 1. Linear warmup scheduler
    linear_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,  # 시작 lr = start_factor * base_lr
        end_factor=1.0,                    # 종료 lr = end_factor * base_lr
        total_iters=warmup_epochs
    )
    
    # 2. Cosine annealing with warm restarts scheduler
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=first_cycle_epochs,  # 첫 번째 cycle의 길이
        T_mult=cycle_mult,       # cycle 길이 증가 비율
        eta_min=min_lr          # 최소 learning rate
    )
    
    # 3. Sequential scheduler로 두 스케줄러 연결
    scheduler = SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]  # warmup_epochs 이후 cosine scheduler로 전환
    )
    
    return scheduler



import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupDecayRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_start_lr: float = 0.0,
        gamma: float = 0.5,  # restart시 max_lr 감소 비율
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: optimizer
            num_warmup_steps: warmup steps 수
            first_cycle_steps: 첫 번째 cosine cycle의 길이
            cycle_mult: 다음 cycle 길이 증가 비율
            max_lr: 최대 learning rate
            min_lr: 최소 learning rate
            warmup_start_lr: warmup 시작 learning rate
            gamma: restart시 max_lr 감소 비율 (0.5면 매 restart마다 max_lr이 절반으로 감소)
        """
        self.num_warmup_steps = num_warmup_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        
        super().__init__(optimizer, last_epoch)
        
        # Initialize learning rate to warmup_start_lr
        self.init_lr()
        
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.warmup_start_lr
    
    def get_lr(self):
        # Warmup 기간
        if self.last_epoch < self.num_warmup_steps:
            return [self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * 
                   (self.last_epoch / self.num_warmup_steps) for _ in self.base_lrs]
        
        # Warmup 이후
        epoch = self.last_epoch - self.num_warmup_steps
        self.step_in_cycle = epoch % self.cur_cycle_steps
        self.cycle = epoch // self.cur_cycle_steps
        
        # Cycle이 바뀌면 max_lr 감소 및 cycle 길이 증가
        if epoch != 0 and self.step_in_cycle == 0:
            self.max_lr = self.max_lr * self.gamma
            self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
        
        # Cosine schedule 계산
        cos_progress = math.pi * (self.step_in_cycle / self.cur_cycle_steps)
        return [self.min_lr + 0.5 * (self.max_lr - self.min_lr) * 
                (1 + math.cos(cos_progress)) for _ in self.base_lrs]