import torch


class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, highest_number=33):
        # NOTE: the y label is not uniformly distributed
        super().__init__()

        self.size = size
        self.highest_number = highest_number

        # generate a synthetic dataset
        # TODO: change this
        torch.manual_seed(42)

        self.x = torch.randint(
            low=0, high=self.highest_number, size=(self.size, 3))
        self.y = torch.sum(self.x, dim=1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.size


class ArithmeticDataset2(torch.utils.data.Dataset):
    def __init__(self, size=1000, highest_number=33, seed=42):
        # NOTE: the y label is not uniformly distributed
        super().__init__()

        self.size = size
        self.highest_number = highest_number
        self.seed = seed

        # generate a synthetic dataset
        torch.manual_seed(self.seed)

        self.x = torch.randint(
            low=0, high=self.highest_number, size=(self.size, 4))
        self.y_T1 = torch.sum(self.x, dim=1)
        self.y_T2 = torch.sum(self.x[:,:3], dim=1)

    def __getitem__(self, index):
        return self.x[index], self.y_T1[index], self.y_T2[index]

    def __len__(self):
        return self.size


if __name__ == '__main__':
    ds = ArithmeticDataset(size=1000, highest_number=33)
    print(ds[5])
    print(ds[6])
    print(ds[7])

    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    print(next(iter(dl)))
