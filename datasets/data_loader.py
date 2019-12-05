import torch.utils.data

class MyDataLoader():

    def __init__(self, dataset, loader_conf, phase='train'):
        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=loader_conf['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=int(loader_conf['num_workers']),
            drop_last=True
        )
    def get_loader(self):
        return self.data_loader
    def __len__(self):
        return len(self.dataset)
