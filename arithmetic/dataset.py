import torch

class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, amount=1000, high=33):
        # NOTE: the y label is not uniformly distributed
        super().__init__()

        self.amount = amount

        # generate a synthetic dataset
        torch.manual_seed(42)
        # self.x = torch.randint(low=0, high=high, size=(amount,3)).float()
        self.x = torch.randint(low=0, high=high, size=(amount,3))
        self.y = torch.sum(self.x,dim=1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.amount

if __name__ == '__main__':
    ds = ArithmeticDataset(1000)
    print(ds[5])
    print(ds[6])
    print(ds[7])

    dl = torch.utils.data.DataLoader(ds, batch_size = 64, shuffle = True)

    print(next(iter(dl)))