# !pip3 install torchsde geomstats

import os

# 1. SET BACKEND FIRST
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'

import torch
import torch.nn as nn
import torch.optim as optim
import torchsde
import numpy as np

# 2. IMPORT GEOMSTATS AFTER
import geomstats.backend as gs
import geomstats.geometry.spd_matrices as spd_matrices

# --- SAFETY CHECK ---
if not torch.is_tensor(gs.array([1., 2.])):
    raise RuntimeError("Geomstats is still using NumPy. RESTART RUNTIME.")
print(f"Geomstats Backend Successfully Loaded: {gs.__name__} (PyTorch)")
# --------------------

gs.set_default_dtype("float32")

# --- 1. Configuration ---
N_ASSETS = 5
STATE_DIM = N_ASSETS * (N_ASSETS + 1) // 2
BATCH_SIZE = 32
TIME_STEPS = 20
N_EPOCHS = 100
LR = 1e-3

# Instantiate the manifold
spd_manifold = spd_matrices.SPDMatrices(n=N_ASSETS)
spd_metric = spd_manifold.metric

IDENTITY = torch.eye(N_ASSETS).float()

print(f"Project: SPD Matrix Forecasting")
print("-" * 30)

# --- 2. Geometric Helper Functions (Robust) ---

def make_spd_safe(matrix, epsilon=1e-5):
    """
    Enforces SPD properties numerically:
    1. Symmetrize: (A + A.T) / 2
    2. Add jitter to diagonal: A + epsilon * I
    """
    # Symmetrize
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    # Add jitter to diagonal
    n = sym.shape[-1]
    jitter = torch.eye(n, device=sym.device).unsqueeze(0) * epsilon
    return sym + jitter

def symm_to_vec(symm_matrix):
    indices = torch.triu_indices(N_ASSETS, N_ASSETS)
    return symm_matrix[:, indices[0], indices[1]]

def vec_to_symm(vec):
    B = vec.shape[0]
    symm_matrix = torch.zeros(B, N_ASSETS, N_ASSETS, device=vec.device)
    indices = torch.triu_indices(N_ASSETS, N_ASSETS)
    symm_matrix[:, indices[0], indices[1]] = vec
    # Symmetrize
    symm_matrix = symm_matrix + symm_matrix.transpose(-1, -2) - torch.diag_embed(torch.diagonal(symm_matrix, dim1=-2, dim2=-1))
    return symm_matrix

def log_map_project(C):
    # Ensure C is strictly SPD before logging
    C_safe = make_spd_safe(C)
    symm_matrix = spd_metric.log(C_safe, base_point=IDENTITY)
    return symm_to_vec(symm_matrix)

def exp_map_wrap(vec):
    symm_matrix = vec_to_symm(vec)
    # Exponentiate to get SPD candidate
    C_candidate = torch.linalg.matrix_exp(symm_matrix)
    # Apply safety guardrail to ensure it stays strictly SPD for the metric
    return make_spd_safe(C_candidate)

# --- 3. The Neural SDE Model ---

class ManifoldSDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'diagonal'

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.state_dim = state_dim
        # Drift
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        # Diffusion
        self.sigma_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()
        )

    def f(self, t, x):
        return self.mu_net(x)

    def g(self, t, x):
        return self.sigma_net(x)

# --- 4. Synthetic Data Generation ---

def get_synthetic_data_path(batch_size, time_steps):
    C_path = torch.zeros(time_steps, batch_size, N_ASSETS, N_ASSETS)
    C_path[0] = IDENTITY.unsqueeze(0).repeat(batch_size, 1, 1)

    current_C = C_path[0]

    for t in range(1, time_steps):
        rand_vec = torch.randn(batch_size, STATE_DIM) * 0.1
        rand_symm = vec_to_symm(rand_vec)
        C_new = torch.linalg.matrix_exp(rand_symm) @ current_C
        C_path[t] = C_new
        current_C = C_new

    # Map entire path to tangent space
    x_path_true = torch.stack([log_map_project(C_path[t]) for t in range(time_steps)])
    return x_path_true.permute(1, 0, 2)

# --- 5. The Training Loop ---

model = ManifoldSDE(state_dim=STATE_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
ts = torch.linspace(0, 1, TIME_STEPS)

print(f"Starting training for {N_EPOCHS} epochs...")

for epoch in range(N_EPOCHS):
    # Generate Data
    x_path_true = get_synthetic_data_path(BATCH_SIZE, TIME_STEPS)
    x0 = x_path_true[:, 0, :]

    optimizer.zero_grad()

    # Solve SDE
    x_path_pred = torchsde.sdeint(model, x0, ts, method='euler', dt=1.0/TIME_STEPS)
    x_path_pred = x_path_pred.permute(1, 0, 2)

    # Reshape for Metric
    B, T, D = x_path_pred.shape

    # Wrap to SPD Manifold (with safety checks inside exp_map_wrap)
    C_path_pred = exp_map_wrap(x_path_pred.reshape(B * T, D))
    C_path_true = exp_map_wrap(x_path_true.reshape(B * T, D))

    # Calculate Loss (Riemannian Distance)
    distances = spd_metric.dist(C_path_pred, C_path_true)
    loss = distances.mean()

    # Backprop
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d} | Geometric Loss: {loss.item():.6f}")

print("Training complete.")

# --- 6. Validation ---

print("\n" + "-" * 30)
print("Starting forecast validation...")
model.eval()

x_path_unseen = get_synthetic_data_path(1, TIME_STEPS)
x0_unseen = x_path_unseen[:, 0, :]
FORECAST_STEPS = 10
ts_forecast = torch.linspace(0, 1, FORECAST_STEPS)
N_SAMPLES = 3

with torch.no_grad():
    x0_expanded = x0_unseen.expand(N_SAMPLES, STATE_DIM)
    x_forecast = torchsde.sdeint(model, x0_expanded, ts_forecast, method='euler', dt=1.0/FORECAST_STEPS)
    x_forecast = x_forecast.permute(1, 0, 2)

    S, T, D = x_forecast.shape
    C_forecast_dist = exp_map_wrap(x_forecast.reshape(S * T, D))
    C_forecast_dist = C_forecast_dist.reshape(S, T, N_ASSETS, N_ASSETS)

    print("Forecast complete.")
    print(f"Forecast shape: {C_forecast_dist.shape} (Samples, Steps, N, N)")
    print("\nLast forecasted matrix (Sample 0):")
    print(C_forecast_dist[0, -1, :, :])

