#Required dependencies and libraries
import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from datetime import datetime
from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
#Import from src
from src.models import MLP
from src.dynamics import Quadrotor2DLearnedDynamics
from src.mpc import MPC
from src.utils import get_pb_handles, get_mass
from src.disturbances import get_wind_function

#Pass argument for different disturbances
import argparse
parser = argparse.ArgumentParser()

parser.add_argument(
    "--wind",
    type=str,
    default="linear"
    choices=["linear", "periodic", "step", "polynomial_like"],
    help="Type of wind disturbance"
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed"
)

args = parser.parse_args()

WIND_TYPE = args.wind
wind_fn = get_wind_function(WIND_TYPE)
print("[DEBUG] wind function =", wind_fn.__name__)

NUM_RUNS = 1
BASE_SEED = args.seed

all_run_errors = []

COST = 'LINEAR_LS'  # NONLINEAR_LS
SAVE_FLAG = True
np.random.seed(42)



    
# ------------------------------------------------------------------------------
# Parameters
# Configure environment parameters
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
print(env.L)

obs, info = env.reset()
xt = obs[:6]  # [pos[0], vel[0], pos[2], vel[2], rpy[1], ang_v[1]]

import pybullet as pb


# ------------------------------------------------------------------------------
# Residual MLP: lightweight
residual_mlp = MLP(input_dim=6 + 2, output_dim=3, hidden_dim=128, num_layers=3)
for param in residual_mlp.parameters():
    param.requires_grad = False
l4c_residual = l4c.L4CasADi(residual_mlp, name="residual_quadrotor2D", mutable=True)

# ------------------------------------------------------------------------------
# MPC Setup
N = 20
t_horizon = 1
learned_model = Quadrotor2DLearnedDynamics(env, l4c_residual)
solver = MPC(model=learned_model.model(), N=N, t_horizon=t_horizon,
                external_shared_lib_dir=l4c_residual.shared_lib_dir,
                external_shared_lib_name=l4c_residual.name).solver

# ------------------------------------------------------------------------------
# Simulation Setup
dt = 1.0 / env_config['pyb_freq']  # Time step
Tsim = 20
Steps = int(Tsim / dt)
x_history, u_history, x_ref_history, opt_times = [xt], [], [], []

for run_id in range(NUM_RUNS):
    

    print(f"\n========== RUN {run_id} ==========\n")

    seed = BASE_SEED + run_id

    # 1️⃣ 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    try:
        pb.resetSimulation()
    except Exception:
        pass


    # 2️⃣ 重建环境
    env_config['seed'] = seed
    env = Quadrotor(**env_config)
    obs, info = env.reset()

    cid, ROBOT_ID = get_pb_handles(env)
    MASS = get_mass(cid, ROBOT_ID)

    print("pybullet bodies:", pb.getNumBodies(physicsClientId=cid))
    print("ROBOT_ID used:", ROBOT_ID)
    print("dyn info exists?", pb.getDynamicsInfo(ROBOT_ID, -1, physicsClientId=cid)[0])
    print("[DEBUG] cid =", cid, "robot_id =", ROBOT_ID, "num_bodies =", pb.getNumBodies(physicsClientId=cid), "mass =", MASS)

    xt = obs[:6]

    x_history = [xt]
    u_history = []
    x_ref_history = []
    Ax=[]

    for i in range(Steps):
        print(f"[STEP {i}] Solving MPC...", flush=True)
        current_time = i * dt

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

        start = time.time()
        # Apply current state as constraint
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        # Solve MPC and apply control
        status=solver.solve()
        ut = solver.get(0, "u")
        u_history.append(ut)


        ax = wind_fn(current_time)
        Ax.append(ax)
        Fx = MASS * ax  # F = m * a
        if i % 50 == 0:  # 每 1 秒打印一次
            base_pos, base_orn = pb.getBasePositionAndOrientation(ROBOT_ID, physicsClientId=cid)
            base_vel, base_avel = pb.getBaseVelocity(ROBOT_ID, physicsClientId=cid)
            print(f"[DEBUG] t={current_time:.2f}, Fx={Fx:.3e}, vx={base_vel[0]:.3f}, x={base_pos[0]:.3f}")

        pb.applyExternalForce(objectUniqueId=ROBOT_ID, linkIndex=-1,
                      forceObj=[Fx, 0.0, 0.0], posObj=[0.0, 0.0, 0.0],
                      flags=pb.WORLD_FRAME,
                      physicsClientId=cid)

        
        # Since simulation_step seems to be missing, let's use the environment step
        next_obs, reward, done, info = env.step(ut)
        xt = next_obs[:6]

        # Add measurement noise (now with correct size=4)
        xt_measured = xt + np.random.normal(0, 0.01, size=6)
        #xt = xt_measured
        
        x_history.append(xt)
        
        if status != 0:
           print("[ACADOS FAIL]", i, "status", status, "u", ut)
           break

        
        
        elapsed = time.time() - start
        opt_times.append(elapsed)
        
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
        csv_path = f"results_stabilization/nominal_{WIND_TYPE}_seed{seed}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved trajectory to {csv_path}")


# Convert to numpy arrays for easier indexing
x_history = np.array(x_history)
u_history = np.array(u_history)
x_ref_history = np.array(x_ref_history)

# Create time grids with matching dimensions
t_grid_states = np.linspace(0, Tsim, len(x_history))
t_grid_inputs = np.linspace(0, Tsim, len(u_history))

print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')
print(f'State history shape: {x_history.shape}, Control history shape: {u_history.shape}')

# ------------------------------------------------------------------------------
# Plot

plt.figure(figsize=(15, 10))
t_grid_states = np.arange(len(x_history)) * dt
winds = [wind_fn(t) for t in t_grid_states]
plt.plot(winds, color='C0', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Wind Accel [m/s²]')
plt.title('Slowly Varying Wind')
plot_path_wind = f"output_new/wind_standard.png"
plt.savefig(plot_path_wind, dpi=300)
plt.grid(True)
plt.tight_layout()

fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True)

labels = ['x [m]', 'x_dot [m/s]',
          'z [m]', 'z_dot [m/s]',
          'theta [rad]', 'theta_dot [rad/s]']

t = np.arange(len(x_history))  # 或 t * dt

for i in range(6):
    axs[i].plot(t, x_history[:, i], linewidth=2)
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)

axs[-1].set_xlabel("Time step")
plt.suptitle("State Trajectories")
plt.tight_layout()

plt.figure(figsize=(15,10))
plt.plot(u_history)
plt.title("u_history")

plt.figure(figsize=(15, 10))

x_err = x_history[1:, 0] - x_ref_history[:, 0]  # 250
z_err = x_history[1:, 2] - x_ref_history[:, 1]  # 250
xz_err = np.sqrt(x_err**2 + z_err**2)

t_err = t_grid_states[1:] 

# Plot x error
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

mean_avg_error = np.mean(xz_err)
print("Mean average X-Z error:", mean_avg_error)

if SAVE_FLAG:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("output", exist_ok=True)
    plot_path = f"output_new/{timestamp}_nominal_standard.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved plot to {plot_path}")

plt.show()

'''
# ------------------------------------------------------------------------------
# Save Results
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
    csv_path = f"results_new/nominal_seed{seed}.csv"
    #csv_path = f"results/known_seed{seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved trajectory to {csv_path}")
'''