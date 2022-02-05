import torch

class CausalArithmetic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Identity()
        self.y = torch.nn.Identity()
        self.z = torch.nn.Identity()

        self.S = torch.nn.Identity()
        self.O = torch.nn.Identity()

    def forward(self,input):
        # We multiply each intermediate value with a trivial layer
        # This allows us to attach hooks to the inttermediate values

        # NOTE: would like to abstract this
        # NOTE: overhead due to pytorch automatically tracking the backward graph,
        # while this is not needed because the model won't be trained?

        # NOTE: without copying, the intervention also changes the input tensor if we intervene on x,y, or z.
        x = torch.clone(input[:,0])
        y = torch.clone(input[:,1])
        z = torch.clone(input[:,2])

        x = self.x(x)
        y = self.y(y)
        z = self.z(z)

        S = self.S(x + y)
        O = self.O(S + z)
        return O
