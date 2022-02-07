import torch

class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, highest_number=33):
        # NOTE: the y label is not uniformly distributed
        super().__init__()

        self.size = size
        self.highest_number = highest_number

        # generate a synthetic dataset
        torch.manual_seed(42)
        
        self.x = torch.randint(low=0, high=self.highest_number, size=(self.size,3))
        self.y = torch.sum(self.x,dim=1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size

if __name__ == '__main__':
    ds = ArithmeticDataset(size=1000, highest_number=33)
    print(ds[5])
    print(ds[6])
    print(ds[7])

    dl = torch.utils.data.DataLoader(ds, batch_size = 64, shuffle = True)

    print(next(iter(dl)))