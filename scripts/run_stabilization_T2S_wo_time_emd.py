import os
import sys
import time
from datetime import datetime

import casadi as cs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybullet as pb

import torch
import torch.nn as nn
import l4casadi as l4c

from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models import TwoSpeedMLP, time_embedding_np
from src.dynamics import Quadrotor2DDynamics
from src.mpc import MPC
from src.utils import get_pb_handles, get_mass
from src.disturbances import get_wind_function
#Pass argument for different disturbances
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--wind",type=str,default="linear",choices=["linear", "periodic", "step", "polynomial_like"],help="Type of wind disturbance")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--A", type=float)
parser.add_argument("--period", type=float)
parser.add_argument("--phase", type=float)
parser.add_argument("--drift_rate", type=float)
parser.add_argument("--slope", type=float)
parser.add_argument("--noise_std", type=float)

args = parser.parse_args()

WIND_TYPE = args.wind
wind_fn = get_wind_function(WIND_TYPE)

wind_params = {}
if args.A is not None:
    wind_params["A"] = args.A
if args.period is not None:
    wind_params["period"] = args.period
if args.phase is not None:
    wind_params["phase"] = args.phase
if args.drift_rate is not None:
    wind_params["drift_rate"] = args.drift_rate
if args.slope is not None:
    wind_params["slope"] = args.slope
if args.noise_std is not None:
    wind_params["noise_std"] = args.noise_std

print("[DEBUG] wind function =", wind_fn.__name__, wind_params)

NUM_RUNS = 10
BASE_SEED = args.seed
TIME_FEAT_DIM = 0 #set to 0 here to evaluate contribution of time embedding
TIME_SCALE = 1.0

all_run_errors = []

COST = 'LINEAR_LS'  # NONLINEAR_LS
SAVE_FLAG = True

env_config = {
    'gui': False,  # Set to False for faster data collection
    'ctrl_freq': 50,  # Control frequency
    'pyb_freq': 50,  # Physics simulation frequency
    'seed': 42,
    'done_on_out_of_bound': True,  # Set to False if you want longer episodes
    
    'init_state_randomization_info': {
        'init_x': {'distrib': 'uniform', 'low': -1, 'high': 1},
        'init_x_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
        'init_z': {'distrib': 'uniform', 'low': 0.5, 'high': 1.5},
        'init_z_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1},
        'init_theta': {'distrib': 'uniform', 'low': -0.2, 'high': 0.2},
        'init_theta_dot': {'distrib': 'uniform', 'low': -0.1, 'high': 0.1}
    }

}

env = Quadrotor(**env_config)
model = env.symbolic

def make_batch_from_samples(samples_np, nominal_func, time_feat_dim):
    """
    samples_np: shape (B, 8 + d + 3)
      first (8+d): network input feature = [x(6), u(2), phi(d)]
      last 3: measured acc [dx2, dz2, dtheta2]

    returns:
      X_batch: torch tensor, shape (B, 8+d)
      y_target: torch tensor, shape (B, 3)
    """
    X_np = samples_np[:, :8 + time_feat_dim].astype(np.float32)
    y_true_np = samples_np[:, 8 + time_feat_dim:].astype(np.float32)

    # use only x,u to compute nominal acceleration
    xu_np = X_np[:, :8]      # [x(6), u(2)]
    x_np = xu_np[:, :6]
    u_np = xu_np[:, 6:8]

    nominal = np.array([
        nominal_func(x_np[j], u_np[j]).full().flatten()
        for j in range(len(samples_np))
    ])

    y_nominal = nominal[:, [1, 3, 5]].astype(np.float32)
    y_target_np = y_true_np - y_nominal

    X_batch = torch.tensor(X_np, dtype=torch.float32)
    y_target = torch.tensor(y_target_np, dtype=torch.float32)

    return X_batch, y_target

def reset_module(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

# MPC Setup
N = 20
t_horizon = 1

# ------------------------------------------------------------------------------
# Simulation Setup
dt = 1.0 / env_config['pyb_freq']
Tsim = 20
Steps = int(Tsim / dt)
#T2S Hyperparameter
B_FAST = 32
B_SLOW = 128
REPLAY_MAX = 5000
SLOW_UPDATE_EVERY = 25
FAST_UPDATE_EVERY = 10

for run_id in range(NUM_RUNS):
    

    print(f"\n========== RUN {run_id} ==========\n")

    seed = BASE_SEED + run_id

    # 1️⃣ Setup random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        pb.resetSimulation()
    except Exception:
        pass


    # 2️⃣ Reset environment
    env_config['seed'] = seed
    env = Quadrotor(**env_config)
    obs, info = env.reset()

    cid, ROBOT_ID = get_pb_handles(env)
    MASS = get_mass(cid, ROBOT_ID)

    print("pybullet bodies:", pb.getNumBodies(physicsClientId=cid))
    print("ROBOT_ID used:", ROBOT_ID)
    print("dyn info exists?", pb.getDynamicsInfo(ROBOT_ID, -1, physicsClientId=cid)[0])
    print("[DEBUG] cid =", cid, "robot_id =", ROBOT_ID, "num_bodies =", pb.getNumBodies(physicsClientId=cid), "mass =", MASS)
    
    dummy_mlp = TwoSpeedMLP(
        input_dim=8 + TIME_FEAT_DIM,
        hidden_dim=64,
        output_dim=3
    )
    dummy_mlp.apply(reset_module)
    del dummy_mlp
    
    # 3️⃣ Reset Neural Network/Solver
    # Create PyTorch residual model
    residual_mlp = TwoSpeedMLP(
        input_dim=8 + TIME_FEAT_DIM,
        hidden_dim=64,
        output_dim=3
    )

    #Build L4CasADi wrapper
    with torch.no_grad():
        dummy_inp = torch.zeros((1, 8 + TIME_FEAT_DIM), dtype=torch.float32)
        l4c_residual = l4c.L4CasADi(
            residual_mlp,
            name="residual_quadrotor2D",
            mutable=True
        )
        l4c_residual.build(inp=dummy_inp)
    for param in residual_mlp.parameters():
        param.requires_grad = False

    learned_model = Quadrotor2DDynamics(env,residual_model=l4c_residual,use_residual=True,use_time_embedding=True,time_feat_dim=TIME_FEAT_DIM,
time_scale=TIME_SCALE,)
    casadi_model = learned_model.model()
    nominal_func = cs.Function('nom', [casadi_model.x, casadi_model.u], [casadi_model.f_nominal])
    solver = MPC(
        model=learned_model.model(),
        N=N,
        t_horizon=t_horizon,
        external_shared_lib_dir=l4c_residual.shared_lib_dir,
        external_shared_lib_name=l4c_residual.name
    ).solver


    #Setup optimizers / loss
    slow_params = list(residual_mlp.layer1.parameters()) + list(residual_mlp.layer2.parameters())
    fast_params = list(residual_mlp.layer3.parameters())

    slow_optimizer = torch.optim.Adam(slow_params, lr=1e-2)
    fast_optimizer = torch.optim.Adam(fast_params, lr=1e-3)
    criterion = nn.MSELoss()

    #Pre-define
    xt = obs[:6]
    obs_buffer = []
    x_history = [xt]
    u_history = []
    x_ref_history = []
    opt_times=[]
    opt_times_fast = []
    opt_times_slow=[]
    loss_fast_history = []
    loss_slow_history = []
    Ax=[]
    x_prev=None

    #Simulation
    for i in range(Steps):
        print(f"[STEP {i}] Solving MPC...", flush=True)

        current_time = i * dt
        for k in range(N):
            t_future = current_time + k * (t_horizon / N)
            tau_k = t_future / TIME_SCALE
            solver.set(k, "p", np.array([tau_k], dtype=np.float64))

        # Set reference for each step in MPC horizon (all zeros)
        for k in range(N):
            if k == 0:  # Only store the first reference of each MPC horizon
                x_ref_history.append([0, 1, 0])
                
            # Set reference for state [x, x_dot, z, z_dot, theta, theta_dot, u1, u2]
            y_ref_k = np.array([0, 0, 1, 0, 0, 0, 0, 0])  # All zeros for reference
            solver.set(k, "yref", y_ref_k)
        # Set terminal reference (only state)
        y_ref_terminal = np.array([0, 0, 1, 0, 0, 0])  # All zeros for terminal reference
        solver.set(N, "yref", y_ref_terminal)

        # Apply current state as constraint
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        # Solve MPC and apply control
        solver.solve()
        ut = solver.get(0, "u")
        u_history.append(ut)

        #Disturbance induce
        ax = wind_fn(current_time, **wind_params)
        Ax.append(ax)
        Fx = MASS * ax  # F = m * a
        if i % 50 == 0:  
            base_pos, base_orn = pb.getBasePositionAndOrientation(ROBOT_ID, physicsClientId=cid)
            base_vel, base_avel = pb.getBaseVelocity(ROBOT_ID, physicsClientId=cid)
            print(f"[DEBUG] t={current_time:.2f}, Fx={Fx:.3e}, vx={base_vel[0]:.3f}, x={base_pos[0]:.3f}")

        pb.applyExternalForce(objectUniqueId=ROBOT_ID, linkIndex=-1,
                      forceObj=[Fx, 0.0, 0.0], posObj=[0.0, 0.0, 0.0],
                      flags=pb.WORLD_FRAME,
                      physicsClientId=cid)
        
        x_prev = xt.copy()
        # Since simulation_step seems to be missing, let's use the environment step
        next_obs, reward, done, info = env.step(ut)
        xt = next_obs[:6]
        
        # Add measurement noise (now with correct size=4)
        xt_measured = xt + np.random.normal(0, 0.01, size=6)
        #xt = xt_measured
        x_history.append(xt)
        
        # --------------------Residual----------------------#
        # Compute x_ddot theta_ddot from difference
        if i > 0:
            
            dx2 = (x_history[-1][1] - x_history[-2][1]) / dt
            dz2 = (x_history[-1][3] - x_history[-2][3]) / dt
            dtheta2 = (x_history[-1][5] - x_history[-2][5]) / dt  # theta_ddot

            # Append: [state (6), action (2), measured accel (3)]
            tau_t = (i * dt) / TIME_SCALE
            phi = time_embedding_np(tau_t, d=TIME_FEAT_DIM)         # (d,)
            
            xu = np.concatenate([x_prev, ut]).astype(np.float32)    # (8,)
            feat = np.concatenate([xu, phi]).astype(np.float32)     # (8+d,)
            
            obs_buffer.append((*feat, dx2, dz2, dtheta2))
            if len(obs_buffer) > REPLAY_MAX:
                obs_buffer.pop(0)
        #fast update
        updated = False
        if i > 0 and (i % FAST_UPDATE_EVERY== 0)and len(obs_buffer) >= B_FAST:
            start = time.time()
            data_fast = np.array(obs_buffer[-B_FAST:], dtype=np.float32)

            # Split into input features and target
            X_fast, y_fast = make_batch_from_samples(
                data_fast,
                nominal_func=nominal_func,
                time_feat_dim=TIME_FEAT_DIM
            )
            
            for p in residual_mlp.layer1.parameters(): p.requires_grad = False
            for p in residual_mlp.layer2.parameters(): p.requires_grad = False
            for p in residual_mlp.layer3.parameters(): p.requires_grad = True
            
            for _ in range(10):
                fast_optimizer.zero_grad()                                  
                pred_fast = residual_mlp(X_fast)
                loss_fast = criterion(pred_fast, y_fast)
                loss_fast.backward()
                fast_optimizer.step()
                loss_fast_history.append(float(loss_fast.detach().cpu()))
                print(loss_fast,"loss_fast")
                updated = True
            
            elapsed_fast = time.time() - start
            print(elapsed_fast, "iteration time")
            opt_times_fast.append(elapsed_fast)

        #slow update
        if i > 0 and (i % SLOW_UPDATE_EVERY == 0) and len(obs_buffer) >= B_SLOW:
            start_2 = time.time()
            idx = np.random.choice(len(obs_buffer), size=B_SLOW, replace=False)
            data_slow = np.array([obs_buffer[j] for j in idx], dtype=np.float32)
            X_slow, y_slow = make_batch_from_samples(
                data_slow,
                nominal_func=nominal_func,
                time_feat_dim=TIME_FEAT_DIM
            )

        # update slow layers
            for p in residual_mlp.layer1.parameters(): p.requires_grad = True
            for p in residual_mlp.layer2.parameters(): p.requires_grad = True
            for p in residual_mlp.layer3.parameters(): p.requires_grad = False

            for _ in range (20):
                slow_optimizer.zero_grad(set_to_none=True)
                for p in residual_mlp.layer3.parameters():
                    p.grad = None
                pred_slow = residual_mlp(X_slow)
                loss_slow = criterion(pred_slow, y_slow)
                print(loss_slow,"loss_slow")
                loss_slow_history.append(float(loss_slow.detach().cpu()))
                loss_slow.backward()
                slow_optimizer.step()
                updated = True
        
            elapsed_slow = time.time() - start_2
            print(elapsed_slow, "iteration time")
            opt_times_slow.append(elapsed_slow)
            
        if updated:
            l4c_residual.update(residual_mlp)
        for p in residual_mlp.parameters():
            p.requires_grad = True
    
    env.close()
    u_history = np.array(u_history)

    # Create time grids with matching dimensions
    t_grid_states = np.linspace(0, Tsim, len(x_history))
    t_grid_inputs = np.linspace(0, Tsim, len(u_history))
    x_history = np.array(x_history) 
    x_ref_history = np.array(x_ref_history)

    x_history = np.array(x_history) 
    x_ref_history = np.array(x_ref_history)
    x_err = x_history[1:, 0] - x_ref_history[:, 0] 
    z_err = x_history[1:, 2] - x_ref_history[:, 1]

    xz_err = np.sqrt(x_err**2 + z_err**2)

    all_run_errors.append(xz_err)

    if SAVE_FLAG:
        min_length = min(len(t_grid_inputs), len(x_history)-1)

        x = x_history[:min_length, 0]
        x_dot = x_history[:min_length, 1]
        z = x_history[:min_length, 2]
        z_dot = x_history[:min_length, 3]
        theta = x_history[:min_length, 4]
        theta_dot = x_history[:min_length, 5]
        u_data = u_history[:min_length]
        x_ref_data = x_ref_history[:min_length, 0]
        z_ref_data = x_ref_history[:min_length, 1]
        theta_ref_data = x_ref_history[:min_length, 2]
        time_data = t_grid_inputs[:min_length]

        df = pd.DataFrame({
            "time": time_data,
            "x": x,
            "x_dot": x_dot,
            "z": z,
            "z_dot": z_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "u1": u_data[:, 0],
            "u2": u_data[:, 1],
            "x_ref": x_ref_data,
            "z_ref": z_ref_data,
            "theta_ref": theta_ref_data
        })

        seed = env_config['seed']
        os.makedirs("results_stabilization", exist_ok=True)

        name_parts = [f"T2S_wo_time_emd_{WIND_TYPE}"]

        if args.A is not None:
            name_parts.append(f"A{args.A}")
        if args.period is not None:
            name_parts.append(f"period{args.period}")
        if args.phase is not None:
            name_parts.append(f"phase{args.phase}")
        if args.drift_rate is not None:
            name_parts.append(f"drift{args.drift_rate}")
        if args.slope is not None:
            name_parts.append(f"slope{args.slope}")
        if args.noise_std is not None:
            name_parts.append(f"noise{args.noise_std}")

        name_parts.append(f"seed{seed}")

        filename = "_".join(name_parts) + ".csv"
        csv_path = os.path.join("results_stabilization", filename)

        df.to_csv(csv_path, index=False)
        print(f"Saved trajectory to {csv_path}")


# Convert to numpy arrays for easier indexing
x_history = np.array(x_history)
u_history = np.array(u_history)
x_ref_history = np.array(x_ref_history)

# Create time grids with matching dimensions
t_grid_states = np.linspace(0, Tsim, len(x_history))
t_grid_inputs = np.linspace(0, Tsim, len(u_history))

#Computation efficiency
print(f'fast iteration time: {1000*np.mean(opt_times_fast):.1f}ms -- {1/np.mean(opt_times_fast):.0f}Hz)')
print(f'slow iteration time: {1000*np.mean(opt_times_slow):.1f}ms -- {1/np.mean(opt_times_slow):.0f}Hz)')
print(f'State history shape: {x_history.shape}, Control history shape: {u_history.shape}')

# ------------------------------------------------------------------------------
# Plot
plt.figure(figsize=(15, 10))
t_grid_states = np.arange(len(x_history)) * dt
plt.plot(Ax, color='C0', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Wind Accel [m/s²]')
plt.title('Wind-induced disturbances')
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(15, 10))
plt.plot(loss_fast_history, color='C0', linewidth=2)
plt.title('Loss Fast')

plt.figure(figsize=(15, 10))
plt.plot(loss_slow_history, color='C0', linewidth=2)
plt.title('Loss Slow')


x_err = x_history[1:, 0] - x_ref_history[:, 0]  # 250
z_err = x_history[1:, 2] - x_ref_history[:, 1]  # 250
xz_err = np.sqrt(x_err**2 + z_err**2)

t_err = t_grid_states[1:] 

mean_avg_error = np.mean(xz_err)
print("Mean average X-Z error:", mean_avg_error)
# Plot x error

plt.figure(figsize=(15, 10))
Y_LIM = (0.0, 0.5)

plt.subplot(3, 1, 1)
plt.plot(t_err, np.abs(x_history[1:, 0]-x_ref_history[:, 0]), linewidth=2)
plt.ylabel('x [m]')
plt.ylim(*Y_LIM)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_err, np.abs(x_history[1:, 2]-x_ref_history[:, 1]), linewidth=2)
plt.ylabel('z [m]')
plt.ylim(*Y_LIM)
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_err, xz_err, linewidth=2)
plt.ylabel('X–Z [m]')
plt.ylim(*Y_LIM)
plt.grid()

plt.figure(figsize=(15, 10))
plt.plot(x_history[:, 0], x_history[:, 2], label="Trajectory", color='C1', linewidth=2)
plt.title('Trajectory')


plt.tight_layout()

plt.show()