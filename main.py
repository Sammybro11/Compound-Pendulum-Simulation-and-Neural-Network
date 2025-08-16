import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import Models
import Simulation as Sim
import time as time
debug = False

device = "mps" if torch.mps.is_available() else "cpu"
if debug: print(device)

## Creating Training Data

phi_init = np.pi/6
phi_dot_init = 0
gamma = 5.0
length = 3.0
radius = 0.3

input_time, input_coords = Sim.Lagrangian_Solver(phi_init, phi_dot_init, gamma, length, radius)
input_States = torch.from_numpy(np.stack([input_coords[0], input_coords[1], input_time],axis = 1)).float().to(device)

lnn = Models.LNN_SoftPlus().to(device)

window_size = 1   # 1-step prediction (can generalize to more)
inputs  = input_States[:-window_size]     # All states upto second last
targets = input_States[window_size:, :2]  # next [phi, phi_dot]
if debug: print(inputs, inputs.shape, targets, targets.shape)
## Shuffling Inputs

perm = torch.randperm(inputs.size(0))  # Generates a random permutation of indices
inputs_shuffled = inputs[perm]
targets_shuffled = targets[perm]

optimizer = optim.Adam(lnn.parameters(), lr=1e-3)
epochs = 200

dt = 50/(np.shape(input_time)[0])
if debug: print(dt)
if debug: print(inputs_shuffled, targets_shuffled)

loss_fn = nn.MSELoss()

for epoch in range(epochs):
    start = time.time()
    epoch_loss = 0.0
    for i in range(len(inputs)):
        phi = inputs_shuffled[i][0]
        phi_dot = targets_shuffled[i][1]
        t = inputs_shuffled[i][2]
        target = targets_shuffled[i]
        phi_ddot = Models.Euler_Lagrange(phi, phi_dot, t, lnn)
        phi_dot_pred = phi_dot + phi_ddot * dt
        phi_pred = phi_dot + phi_dot_pred * dt

        state_pred = torch.stack([phi_pred, phi_dot_pred], dim = 0)

        loss = loss_fn(state_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    end = time.time()
    print(f'epoch: {epoch}, loss: {epoch_loss/ len(inputs):.6f}, time: {end-start:.6f}')

print("Training complete.")





