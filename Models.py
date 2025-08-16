
import torch
import torch.nn as nn
import torch.optim as optim

class LNN_SoftPlus(nn.Module):
    def __init__(self):
        super(LNN_SoftPlus, self).__init__()
        self.Linear_Stack = nn.Sequential(
            nn.Linear(3, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 1)
        )

    def forward(self, phi, phi_dot, t):
        state = torch.cat([phi.unsqueeze(0), phi_dot.unsqueeze(0), t.unsqueeze(0)], dim=-1)
        return self.Linear_Stack(state).squeeze(-1)


def Euler_Lagrange(phi, phi_dot, t, L_fn):
    phi.requires_grad_()
    phi_dot.requires_grad_()
    t.requires_grad_()
    L = L_fn(phi, phi_dot, t)
    grad_phi = torch.autograd.grad(L, phi, create_graph= True, retain_graph=True)[0]
    grad_phi_dot = torch.autograd.grad(L, phi_dot,  create_graph= True, retain_graph=True)

    Hess = torch.autograd.functional.hessian(
        lambda pd: L_fn(phi, pd, t), phi_dot,  create_graph= True)
    Jaco = torch.autograd.functional.jacobian(
        lambda p: torch.autograd.grad(L_fn(p, phi_dot, t), phi_dot, create_graph= True, retain_graph= True)[0],
        phi, create_graph=True)

    TimeJac = torch.autograd.functional.jacobian(
        lambda tt: torch.autograd.grad(L_fn(phi, phi_dot, tt), phi_dot, create_graph=True, retain_graph=True)[0],
        t, create_graph=True
    )


    phi_ddot = ( grad_phi - Jaco * phi_dot - TimeJac)/Hess
    return phi_ddot


class NN(nn.Module):
    def __init__(self):
        super(LNN_SoftPlus, self).__init__()
        self.Linear_Stack = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, phi, phi_dot, gamma , t):
        state = torch.cat([phi, phi_dot, t], dim=-1)
        return self.Linear_Stack(state).squeeze(-1)