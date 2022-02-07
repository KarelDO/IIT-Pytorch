from turtle import forward
import torch

class NeuralArithmetic(torch.nn.Module):
    def __init__(self, onehot_width=None, model_hidden_width=None):
        super().__init__()

        self.onehot_width = onehot_width
        self.model_hidden_width = model_hidden_width

        self.embed = torch.nn.Linear(self.onehot_width, self.model_hidden_width)

        self.ff1 = torch.nn.Linear(3*self.model_hidden_width,3*self.model_hidden_width)
        self.ff2 = torch.nn.Linear(3*self.model_hidden_width,3*self.model_hidden_width)
        self.ff3 = torch.nn.Linear(3*self.model_hidden_width,3*self.onehot_width)

        # self.act1 = torch.nn.ReLU()
        self.act1 = torch.nn.Tanh()
        # self.act2 = torch.nn.ReLU()
        self.act2 = torch.nn.Tanh()

        # some magic to easily access parts of the layers
        self.identity_x = torch.nn.Identity()
        self.identity_y = torch.nn.Identity()
        self.identity_z = torch.nn.Identity()

        self.identity_a = torch.nn.Identity()
        self.identity_b = torch.nn.Identity()
        self.identity_c = torch.nn.Identity()

        self.identity_d = torch.nn.Identity()
        self.identity_e = torch.nn.Identity()
        self.identity_f = torch.nn.Identity()

        self.identity_o = torch.nn.Identity()

    def forward(self, input):
        # input [batch, 3]
        x = input[:,0]
        y = input[:,1]
        z = input[:,2]

        x = self._convert_to_onehot(x)
        y = self._convert_to_onehot(y)
        z = self._convert_to_onehot(z)

        x = self.embed(x)
        y = self.embed(y)
        z = self.embed(z)

        # making the slices of the layers more accessible for interventions
        x,y,z = self.identity_x(x), self.identity_y(y), self.identity_z(z)

        x = torch.cat((x,y,z), dim=1)
        x = self.act1(self.ff1(x))

        # making the slices of the layers more accessible for interventions
        a,b,c = x[:,0:self.model_hidden_width], x[:,self.model_hidden_width:2*self.model_hidden_width], x[:,2*self.model_hidden_width:3*self.model_hidden_width]
        a,b,c = self.identity_a(a), self.identity_b(b), self.identity_c(c)
        x = torch.cat((a,b,c), dim=1)

        x = self.act2(self.ff2(x))

        # making the slices of the layers more accessible for interventions
        d,e,f = x[:,0:self.model_hidden_width], x[:,self.model_hidden_width:2*self.model_hidden_width], x[:,2*self.model_hidden_width:3*self.model_hidden_width]
        d,e,f = self.identity_d(d), self.identity_e(e), self.identity_f(f)
        x = torch.cat((d,e,f), dim=1)

        x = self.ff3(x)

        # making the slices of the layers more accessible for interventions
        x = self.identity_o(x)

        return x

    def _convert_to_onehot(self, input):
        # input [batch, 1]
        onehot = torch.zeros((input.shape[0], self.onehot_width))
        onehot[range(input.shape[0]), input] = 1
        return onehot

