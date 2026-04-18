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

from src.models import MLP
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
parser.add_argument("--quiet", action="store_true", help="Reduce console output")
parser.add_argument("--no-plot", action="store_true", help="Disable all plotting")

args = parser.parse_args()

DO_PLOT = not args.no_plot
VERBOSE = not args.quiet

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

print("wind function =", wind_fn.__name__, wind_params)

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

NUM_RUNS = 1
BASE_SEED = args.seed

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

def draw_overlay(frame, disturbance, current_time, method_name="Nominal MPC"):
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

# MPC Setup
N = 20
t_horizon = 1

# ------------------------------------------------------------------------------
# Simulation Setup
dt = 1.0 / env_config['pyb_freq']
Tsim = 20
Steps = int(Tsim / dt)
batch_size = 32

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

    video_writer = None

    if RECORD_DEMO and run_id == DEMO_RUN_ID:
        os.makedirs("demo_videos", exist_ok=True)

        video_name_parts = [f"Neural_MPC_tracking_figure8_demo_{WIND_TYPE}"]
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

    cid, ROBOT_ID = get_pb_handles(env)
    MASS = get_mass(cid, ROBOT_ID)

    print("pybullet bodies:", pb.getNumBodies(physicsClientId=cid))
    print("ROBOT_ID used:", ROBOT_ID)
    vprint("dyn info exists?", pb.getDynamicsInfo(ROBOT_ID, -1, physicsClientId=cid)[0])
    vprint("[DEBUG] cid =", cid, "robot_id =", ROBOT_ID, "num_bodies =", pb.getNumBodies(physicsClientId=cid), "mass =", MASS)
    
    # 3️⃣ Reset Neural Network/Solver
    residual_mlp = MLP(input_dim=6 + 2, output_dim=3, hidden_dim=64, num_layers=3)
    for param in residual_mlp.parameters():
        param.requires_grad = False
    l4c_residual = l4c.L4CasADi(residual_mlp, name="residual_quadrotor2D", mutable=True)
    residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-3)
    residual_criterion = nn.MSELoss()

    learned_model = Quadrotor2DDynamics(env, residual_model=l4c_residual, use_residual=True, use_time_embedding=False)
    casadi_model = learned_model.model()
    nominal_func = cs.Function('nom', [casadi_model.x, casadi_model.u], [casadi_model.f_nominal])
    solver = MPC(
        model=learned_model.model(),
        N=N,
        t_horizon=t_horizon,
        external_shared_lib_dir=l4c_residual.shared_lib_dir,
        external_shared_lib_name=l4c_residual.name
    ).solver
    
    #Pre-define
    obs_buffer = []
    loss_history = []

    xt = obs[:6]

    x_history = [xt]
    u_history = []
    x_ref_history = []
    Ax=[]
    opt_times =[]
    error_history=[]

    cam_x = None
    cam_z = None
    cam_alpha = 0.9

    #Simulation
    for i in range(Steps):
        vprint(f"[STEP {i}] Solving MPC...", flush=True)
        current_time = i * dt

        # Figure-8 parameters
        A = 0.5
        B = 0.5
        center_x = 0.0
        center_z = 1.0
        omega = 2 * np.pi / Tsim

        for k in range(N):
            t_future = current_time + k * t_horizon / N

            x_ref = center_x + A * np.sin(omega * t_future)
            z_ref = center_z + B * np.sin(omega * t_future)*np.cos(omega*t_future)

            if k == 0:
                x_ref_history.append([x_ref, z_ref, 0])

            y_ref_k = np.array([x_ref, 0, z_ref, 0, 0, 0, 0, 0])
            solver.set(k, "yref", y_ref_k)

        # Set terminal reference (only state)
        t_terminal = current_time + t_horizon
        x_term = center_x + A * np.sin(omega * t_terminal)
        z_term = center_z + B * np.sin(omega * t_terminal)*np.cos(omega*t_terminal)
        y_ref_terminal = np.array([x_term, 0, z_term, 0, 0, 0])
        solver.set(N, "yref", y_ref_terminal)

        # Apply current state as constraint
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        
        start = time.time()
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
            vprint(f"[DEBUG] t={current_time:.2f}, Fx={Fx:.3e}, vx={base_vel[0]:.3f}, x={base_pos[0]:.3f}")

        pb.applyExternalForce(objectUniqueId=ROBOT_ID, linkIndex=-1,
                      forceObj=[Fx, 0.0, 0.0], posObj=[0.0, 0.0, 0.0],
                      flags=pb.WORLD_FRAME,
                      physicsClientId=cid)
        
        # Since simulation_step seems to be missing, let's use the environment step
        next_obs, reward, done, info = env.step(ut)
        xt = next_obs[:6]

        next_time = (i + 1) * dt
        x_ref_next = center_x + A * np.sin(omega * next_time)
        z_ref_next = center_z + B * np.sin(omega * next_time) * np.cos(omega * next_time)

        current_error = np.sqrt((xt[0] - x_ref_next)**2 + (xt[2] - z_ref_next)**2)
        error_history.append(current_error)
        
        # Add measurement noise (now with correct size=4)
        xt_measured = xt + np.random.normal(0, 0.01, size=6)
        #xt = xt_measured
        x_history.append(xt)

        if video_writer is not None:
            if 8.0 <= current_time <= 12.0:
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
                    method_name="Neural MPC"
                )

                video_writer.append_data(frame)

        # --------------------Residual----------------------#
        # Compute x_ddot theta_ddot from difference
        if i > 0:
            dx2 = (x_history[-1][1] - x_history[-2][1]) / dt
            dz2 = (x_history[-1][3] - x_history[-2][3]) / dt
            dtheta2 = (x_history[-1][5] - x_history[-2][5]) / dt  # theta_ddot

            # Append: [state (6), action (2), measured accel (3)]
            obs_buffer.append((*xt, *ut, dx2, dz2, dtheta2))

        if i > 0 and i % int(0.5 / dt) == 0 and len(obs_buffer) >= batch_size:
            start = time.time()
            data = np.array(obs_buffer[-batch_size:])   # Recent data

            X_batch = torch.tensor(data[:, :8], dtype=torch.float32)
            y_true = data[:, 8:]

            nominal = np.array([
                nominal_func(x[:6], x[6:8]).full().flatten() for x in data
            ])
            y_nominal = nominal[:, [1, 3, 5]]

            y_target = torch.tensor(y_true - y_nominal, dtype=torch.float32)

            for p in residual_mlp.parameters(): p.requires_grad = True
            for _ in range (20):
                residual_optimizer.zero_grad()
                pred = residual_mlp(X_batch)
                loss = residual_criterion(pred, y_target)
                loss_history.append(loss.item())

                loss.backward()
                vprint(loss, "loss")
                residual_optimizer.step()

            for p in residual_mlp.parameters():
                p.requires_grad = False

            l4c_residual.update(residual_mlp)
            elapsed = time.time() - start
            opt_times.append(elapsed)
            vprint(elapsed, "iteration time")
                
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
    #print mean error
    mean_avg_error = np.mean(xz_err)
    print("Mean average X-Z error:", mean_avg_error)

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
        os.makedirs("results_tracking_figure8", exist_ok=True)

        name_parts = [f"Neural_MPC_{WIND_TYPE}"]

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
        csv_path = os.path.join("results_tracking_figure8", filename)

        df.to_csv(csv_path, index=False)
        print(f"Saved trajectory to {csv_path}")



# Convert to numpy arrays for easier indexing
x_history = np.array(x_history)
u_history = np.array(u_history)
x_ref_history = np.array(x_ref_history)

# Create time grids with matching dimensions
t_grid_states = np.linspace(0, Tsim, len(x_history))
t_grid_inputs = np.linspace(0, Tsim, len(u_history))

vprint(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')
vprint(f'State history shape: {x_history.shape}, Control history shape: {u_history.shape}')

# ------------------------------------------------------------------------------

vprint(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')
vprint(f'State history shape: {x_history.shape}, Control history shape: {u_history.shape}')

# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(error_history, color='C0', linewidth=2)
plt.title('Current Error')
plt.grid(True)
plt.tight_layout()

plt.show()

'''
# Plot
if DO_PLOT:
    plt.figure(figsize=(10, 4))
    plt.plot(u_history, color='C0', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Thrust')
    plt.title('force')
    plt.grid(True)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))

    plt.plot(Ax, color='C0', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Wind Accel [m/s²]')
    plt.title('Slowly Varying Wind')
    plt.grid(True)
    plt.tight_layout()

    # Plot error curve
    plt.figure(figsize=(10, 4))
    x_err = x_history[1:, 0] - x_ref_history[:, 0]
    z_err = x_history[1:, 2] - x_ref_history[:, 1]

    xz_err = np.sqrt(x_err**2 + z_err**2)


    plt.plot(xz_err, label='Euclidean x-z error', color='C3', linewidth=2, linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Error [m]')
    plt.legend()
    plt.grid()

    # Plot x-z plane (2D trajectory)
    plt.figure(figsize=(8, 8))
    plt.plot(x_history[:, 0], x_history[:, 2], label="Trajectory", color='C1', linewidth=2)
    plt.plot(x_ref_history[:, 0], x_ref_history[:, 1], '--', label="Reference", color='C0', linewidth=2)
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title("2D Trajectory in x-z Plane")
    plt.grid()
    plt.axis('equal')  # Make x and z scales equal
    plt.legend()

    plt.tight_layout()
    
    plt.show()
'''