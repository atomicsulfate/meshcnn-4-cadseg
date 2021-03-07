from meshcnn.data import DataLoader, CreateDataset
import torch.utils.data
from data.base_dataset import collate_fn

class DistributedDataLoader(DataLoader):
    """multi-process data loading"""
    def __init__(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset,
            shuffle=not opt.serial_batches
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            num_workers=int(opt.num_threads),
            collate_fn=collate_fn,
            sampler=self.sampler)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)