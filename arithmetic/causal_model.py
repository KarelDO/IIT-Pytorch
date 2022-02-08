import torch


class CausalArithmetic(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.x = torch.nn.Identity()
        self.y = torch.nn.Identity()
        self.z = torch.nn.Identity()

        self.S = torch.nn.Identity()
        self.O = torch.nn.Identity()

    def forward(self, input):
        # We multiply each intermediate value with a trivial layer
        # This allows us to attach hooks to the inttermediate values

        # NOTE: would like to abstract this
        # NOTE: overhead due to pytorch automatically tracking the backward graph,
        # while this is not needed because the model won't be trained?

        # NOTE: without copying, the intervention also changes the input tensor if we intervene on x,y, or z.
        x = torch.clone(input[:, 0])
        y = torch.clone(input[:, 1])
        z = torch.clone(input[:, 2])

        x = self.x(x)
        y = self.y(y)
        z = self.z(z)

        S = self.S(x + y)
        O = self.O(S + z)
        return O


class CausalArithmetic2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w = torch.nn.Identity()
        self.x = torch.nn.Identity()
        self.y = torch.nn.Identity()
        self.z = torch.nn.Identity()

        self.S1 = torch.nn.Identity()
        self.S2 = torch.nn.Identity()

        self.C1 = torch.nn.Identity()
        self.C2 = torch.nn.Identity()
        self.C3 = torch.nn.Identity()

        self.O = torch.nn.Identity()

    def forward(self, input):
        w = torch.clone(input[:, 0])
        x = torch.clone(input[:, 1])
        y = torch.clone(input[:, 2])
        z = torch.clone(input[:, 3])

        w = self.w(w)
        x = self.x(x)
        y = self.y(y)
        z = self.z(z)

        S1 = self.S1(w + x)
        C1 = self.C1(y)
        C2 = self.C2(z)
        S2 = self.S2(S1 + C1)
        C3 = self.C3(C2)
        O = self.O(S2 + C3)
        return O
