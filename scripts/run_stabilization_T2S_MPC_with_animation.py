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

NUM_RUNS = 1
BASE_SEED = args.seed
TIME_FEAT_DIM = 32
TIME_SCALE = 1.0

all_run_errors = []

COST = 'LINEAR_LS'  # NONLINEAR_LS
SAVE_FLAG = True

# Packages for animation
import cv2
import imageio.v2 as imageio
import matplotlib.cm as cm
import matplotlib.colors as mcolors

DIST_MIN = -0.005
DIST_MAX = 0.005

cmap = cm.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=DIST_MIN, vmax=DIST_MAX)

def disturbance_to_bgr(d):
    rgba = cmap(norm(d))
    rgb = tuple(int(255 * c) for c in rgba[:3])
    return (rgb[2], rgb[1], rgb[0])   # OpenCV is BGR

ERR_MIN = 0.0
ERR_MAX = 0.05   

err_cmap = cm.get_cmap("viridis_r")
err_norm = mcolors.Normalize(vmin=ERR_MIN, vmax=ERR_MAX)

def error_to_bgr(e):
    rgba = err_cmap(err_norm(e))
    rgb = tuple(int(255 * c) for c in rgba[:3])
    return (rgb[2], rgb[1], rgb[0])

def capture_frame(cid, target_x, target_z, width=960, height=720):
    #tracking's capture_frame function
    cam_offset_y = -0.8
    cam_offset_z = 0.12

    eye_x = target_x
    eye_y = cam_offset_y
    eye_z = target_z + cam_offset_z

    view_matrix = pb.computeViewMatrix(
        cameraEyePosition=[eye_x, eye_y, eye_z],
        cameraTargetPosition=[target_x, 0.0, target_z],
        cameraUpVector=[0, 0, 1]
    )

    proj_matrix = pb.computeProjectionMatrixFOV(
        fov=45,
        aspect=width / height,
        nearVal=0.1,
        farVal=10.0
    )

    img = pb.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=cid
    )

    rgba = np.array(img[2], dtype=np.uint8).reshape(height, width, 4)
    frame = rgba[:, :, :3].copy()
    return frame, view_matrix, proj_matrix

def draw_trail(frame, x_history, error_history, view_matrix, proj_matrix, trail_len=40):
    h, w = frame.shape[:2]

    if len(x_history) < 2 or len(error_history) < 1:
        return frame

    num_segments = min(len(x_history) - 1, len(error_history))

    start_idx = max(0, num_segments - trail_len)

    pixel_points = []
    segment_errors = []

    for seg_idx in range(start_idx, num_segments):
        state = x_history[seg_idx + 1]   
        x = state[0]
        z = state[2]

        pt = world_to_pixel([x, 0.0, z], view_matrix, proj_matrix, w, h)
        if pt is not None:
            pixel_points.append(pt)
            segment_errors.append(error_history[seg_idx])

    if len(pixel_points) < 2:
        return frame

    for i in range(1, len(pixel_points)):
        e = segment_errors[i]
        color = error_to_bgr(e)

        alpha = i / len(pixel_points)
        thickness = max(1, int(1 + 4 * alpha))

        cv2.line(
            frame,
            pixel_points[i - 1],
            pixel_points[i],
            color,
            thickness,
            cv2.LINE_AA
        )

    cv2.circle(frame, pixel_points[-1], 5, (0, 255, 255), -1, cv2.LINE_AA)

    return frame


def world_to_pixel(point_world, view_matrix, proj_matrix, width, height):
    """
    point_world: [x, y, z]
    return: (u, v) in image pixel coordinates, or None if out of view
    """
    pw = np.array([point_world[0], point_world[1], point_world[2], 1.0])

    view = np.array(view_matrix).reshape(4, 4, order='F')
    proj = np.array(proj_matrix).reshape(4, 4, order='F')

    clip = proj @ (view @ pw)

    if abs(clip[3]) < 1e-8:
        return None

    ndc = clip[:3] / clip[3]   # normalized device coordinates in [-1,1]


    if ndc[0] < -1 or ndc[0] > 1 or ndc[1] < -1 or ndc[1] > 1:
        return None

    u = int((ndc[0] + 1) * 0.5 * width)
    v = int((1 - (ndc[1] + 1) * 0.5) * height)   
    return (u, v)

def draw_text_with_outline(frame, text, pos, font_scale=0.8,
                           text_color=(255, 255, 255),
                           outline_color=(0, 0, 0),
                           text_thickness=2,
                           outline_thickness=5):
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA
    )
    cv2.putText(
        frame,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )

def draw_overlay(frame, disturbance, current_time, method_name="T2S MPC"):
    h, w = frame.shape[:2]

    color = disturbance_to_bgr(disturbance)

    x0, y0 = 100, h - 90
    max_len = 140
    scale = max(abs(DIST_MIN), abs(DIST_MAX))
    dx = int(np.clip(disturbance / scale, -1, 1) * max_len)

    cv2.arrowedLine(
        frame,
        (x0, y0),
        (x0 + dx, y0),
        color,
        thickness=6,
        tipLength=0.25
    )

    draw_text_with_outline(frame, method_name, (30, 40), 1.0)
    draw_text_with_outline(frame, f"t = {current_time:.2f}s", (30, 80), 0.8)
    draw_text_with_outline(frame,f"disturbance = {disturbance:+.4f} m/s^2", (30, h - 30), 0.8)
    return frame

def draw_colorbar(frame, current_disturbance, dist_min, dist_max, bar_x=None):
    h, w = frame.shape[:2]

    bar_width = 24
    bar_height = 220

    if bar_x is None:
        bar_x = w - 70
    bar_y = 80

    for i in range(bar_height):
        ratio = 1.0 - i / (bar_height - 1)
        value = dist_min + ratio * (dist_max - dist_min)
        color = disturbance_to_bgr(value)

        cv2.line(
            frame,
            (bar_x, bar_y + i),
            (bar_x + bar_width, bar_y + i),
            color,
            1
        )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        1
    )

    draw_text_with_outline(frame, f"{dist_max:+.3f}", (bar_x - 5, bar_y - 10), 0.5,
                       text_color=(0, 255, 255), text_thickness=1, outline_thickness=4)
    draw_text_with_outline(frame, f"{0:+.3f}", (bar_x - 5, bar_y + bar_height // 2 + 5), 0.5,
                        text_color=(0, 255, 255), text_thickness=1, outline_thickness=4)
    draw_text_with_outline(frame, f"{dist_min:+.3f}", (bar_x - 5, bar_y + bar_height + 20), 0.5,
                        text_color=(0, 255, 255), text_thickness=1, outline_thickness=4)
    draw_text_with_outline(frame, "dist", (bar_x - 2, bar_y + bar_height + 45), 0.55,
                        text_color=(0, 255, 255), text_thickness=1, outline_thickness=4)

    ratio = (current_disturbance - dist_min) / (dist_max - dist_min)
    ratio = np.clip(ratio, 0, 1)
    y_marker = int(bar_y + (1 - ratio) * bar_height)

    cv2.line(
        frame,
        (bar_x - 10, y_marker),
        (bar_x + bar_width + 10, y_marker),
        (255, 255, 255),
        2
    )

    cv2.circle(frame, (bar_x + bar_width // 2, y_marker), 4, (255, 255, 255), -1)

    return frame

def draw_error_colorbar(frame, current_error, err_min, err_max, bar_x=None):
    h, w = frame.shape[:2]

    bar_width = 32
    bar_height = 300

    if bar_x is None:
        bar_x = w - 90
    bar_y = 80

    for i in range(bar_height):
        ratio = 1.0 - i / (bar_height - 1)
        value = err_min + ratio * (err_max - err_min)
        color = error_to_bgr(value)

        cv2.line(
            frame,
            (bar_x, bar_y + i),
            (bar_x + bar_width, bar_y + i),
            color,
            1
        )

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (255, 255, 255),
        1
    )

    draw_text_with_outline(frame, f"{err_max:.2f}", (bar_x - 5, bar_y - 10), 0.7,
                       text_color=(255, 255, 255), text_thickness=2, outline_thickness=6)
    draw_text_with_outline(frame, f"{0:.2f}", (bar_x - 5, bar_y + bar_height + 20), 0.7,
                        text_color=(255, 255, 255), text_thickness=2, outline_thickness=6)
    draw_text_with_outline(frame, "error[m]", (bar_x - 20, bar_y + bar_height + 45), 0.7,
                        text_color=(255, 255, 255), text_thickness=2, outline_thickness=6)

    ratio = (current_error - err_min) / (err_max - err_min)
    ratio = np.clip(ratio, 0, 1)
    y_marker = int(bar_y + (1 - ratio) * bar_height)

    cv2.line(
        frame,
        (bar_x - 10, y_marker),
        (bar_x + bar_width + 10, y_marker),
        (255, 255, 255),
        3
    )

    cv2.circle(frame, (bar_x + bar_width // 2, y_marker), 6, (255, 255, 255), -1)

    return frame

RECORD_DEMO = True
DEMO_RUN_ID = 0
VIDEO_FPS = 50

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
SLOW_UPDATE_EVERY = 50
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

    video_writer = None

    if RECORD_DEMO and run_id == DEMO_RUN_ID:
        os.makedirs("demo_videos", exist_ok=True)

        video_name_parts = [f"T2S_MPC_stabilization_demo_{WIND_TYPE}"]
        if args.A is not None:
            video_name_parts.append(f"A{args.A}")
        if args.period is not None:
            video_name_parts.append(f"period{args.period}")
        if args.phase is not None:
            video_name_parts.append(f"phase{args.phase}")
        if args.drift_rate is not None:
            video_name_parts.append(f"drift{args.drift_rate}")
        if args.slope is not None:
            video_name_parts.append(f"slope{args.slope}")
        if args.noise_std is not None:
            video_name_parts.append(f"noise{args.noise_std}")
        video_name_parts.append(f"seed{seed}")

        video_filename = "_".join(video_name_parts) + ".mp4"
        video_path = os.path.join("demo_videos", video_filename)

        video_writer = imageio.get_writer(video_path, fps=VIDEO_FPS, codec="libx264")


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

    error_history=[]

    cam_x = None
    cam_z = None
    cam_alpha = 0.9

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

        current_error = np.sqrt((xt[0] - 0.0)**2 + (xt[2] - 1.0)**2)
        error_history.append(current_error)

        # Add measurement noise (now with correct size=4)
        xt_measured = xt + np.random.normal(0, 0.01, size=6)
        #xt = xt_measured
        x_history.append(xt)

        if video_writer is not None:
            if 16.0 <= current_time <= 20.0:
                base_pos, base_orn = pb.getBasePositionAndOrientation(ROBOT_ID, physicsClientId=cid)
                raw_x = base_pos[0]
                raw_z = base_pos[2]

                if cam_x is None:
                    cam_x = raw_x
                    cam_z = raw_z
                else:
                    cam_x = cam_alpha * cam_x + (1 - cam_alpha) * raw_x
                    cam_z = cam_alpha * cam_z + (1 - cam_alpha) * raw_z

                frame, view_matrix, proj_matrix = capture_frame(
                    cid,
                    target_x=cam_x,
                    target_z=cam_z
                )

                frame = draw_trail(frame, x_history, error_history, view_matrix, proj_matrix, trail_len=40)
                frame = draw_error_colorbar(
                    frame,
                    current_error=current_error,
                    err_min=ERR_MIN,
                    err_max=ERR_MAX
                )
                frame = draw_overlay(
                    frame,
                    disturbance=ax,
                    current_time=current_time,
                    method_name="T2S MPC"
                )

                video_writer.append_data(frame)
        
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

            for p in residual_mlp.parameters():
                p.requires_grad = True

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
    
    if video_writer is not None:
        video_writer.close()
        print(f"Saved demo video to {video_path}")
    
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

        name_parts = [f"T2S_MPC_{WIND_TYPE}"]

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
plt.figure(figsize=(10, 4))
plt.plot(error_history, color='C0', linewidth=2)
plt.title('Current Error')
plt.grid(True)
plt.tight_layout()

plt.show()
'''
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
'''