"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import argparse
import time

import torch
import matplotlib.pyplot as plt

from neurowalker.controllers.cpg.hopf import (
    HopfNetworkControllerCfg,
    HopfNetworkController,
)
from neurowalker.controllers.ik import (
    InverseKinematicsControllerCfg,
    InverseKinematicsController,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example script with random-command flag"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Controller update rate in seconds. Controls the simulation timestep. Smaller values = smoother but more computation. Defaults to 0.02 s",
    )
    parser.add_argument(
        "--integration-method",
        type=str,
        choices=("heun", "rk4"),
        default="heun",
        help="Numerical method for integrating oscillator states. RK4 is more accurate, Heun is faster. Defaults to 'heun'",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=32,
        help="Convergence gain for oscillator phase/amplitude error correction. Higher = faster locking to desired gait. Defaults to 32",
    )
    parser.add_argument(
        "--default-alpha",
        type=float,
        default=(0, math.pi, math.pi, 0, 0, math.pi),
        nargs="+",
        help="Initial oscillator phase offsets. Defaults to (0, math.pi, math.pi, 0, 0, math.pi), e.g. tripod gait of a 6-legged robot",
    )
    parser.add_argument(
        "--mu-min",
        type=float,
        default=0.0,
        help="Minimum allowed amplitude modulation. Defaults to 1.0",
    )
    parser.add_argument(
        "--mu-max",
        type=float,
        default=2.0,
        help="Maximum allowed amplitude modulation. Defaults to 3.0",
    )
    parser.add_argument(
        "--w-min",
        type=float,
        default=0.0,
        help="Minimum oscillator frequency modulation (rad/s). Defaults to 0.0",
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=math.pi,
        help="Maximum oscillator frequency modulation (rad/s). In real scenarios, this is should be modulated externally. Defaults to math.pi",
    )
    parser.add_argument(
        "--omega-cmd-min",
        type=float,
        default=-math.pi,
        help="Minimum robot heading change command (rad/s). Defaults to -math.pi",
    )
    parser.add_argument(
        "--omega-cmd-max",
        type=float,
        default=math.pi,
        help="Maximum robot heading change command (rad/s). Defaults to math.pi",
    )
    parser.add_argument(
        "--omega-cmd-tau",
        type=float,
        default=0.25,
        help="Time constant of heading command low-pass filter. Smooths sharp turn commands. Defaults to 0.25 s",
    )
    parser.add_argument(
        "--self-weight",
        type=float,
        default=0.0,
        help="Coupling weight for an oscillator with itself (self-coupling). Defaults to 0.0",
    )
    parser.add_argument(
        "--in-group-weight",
        type=float,
        default=1.0,
        help="Coupling weight for oscillators in the same group. Defaults to 1.0",
    )
    parser.add_argument(
        "--of-group-weight",
        type=float,
        default=0.0,
        help="Coupling weight for oscillators in different groups. Defaults to 0.0",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum phase difference (in radians) to consider oscillators as belonging to the same group. Defaults to 0.0 rad",
    )
    parser.add_argument(
        "--s-min",
        type=float,
        default=0.05,
        help="inimum leg stride length. Controls how short steps can be when reducing gait amplitude. Default to 0.05 m",
    )
    parser.add_argument(
        "--s-max",
        type=float,
        default=0.1,
        help="Maximum leg stride length. Determines the longest horizontal displacement per step. Default to 0.1 m",
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=0.07,
        help="Minimum body height relative to the ground. Used in crouching or low-clearance modes. Default to 0.07 m",
    )
    parser.add_argument(
        "--h-max",
        type=float,
        default=0.15,
        help="aximum body height. Allows higher stance for better obstacle clearance. Default to 0.15 m",
    )
    parser.add_argument(
        "--d-min",
        type=float,
        default=0.0,
        help="Minimum horizontal displacement of the foot during a single step (step size). Default to 0.0 m",
    )
    parser.add_argument(
        "--d-max",
        type=float,
        default=0.07,
        help="Maximum horizontal displacement of the foot during a single step. Default to 0.07 m",
    )
    parser.add_argument(
        "--g-c-min",
        type=float,
        default=0.0,
        help="Minimum vertical clearance of the foot tip above the ground during swing phase. Default to 0.0 m",
    )
    parser.add_argument(
        "--g-c-max",
        type=float,
        default=0.07,
        help="Maximum vertical clearance of the foot tip above the ground during swing phase. Default to 0.07 m",
    )
    parser.add_argument(
        "--g-p-min",
        type=float,
        default=0.0,
        help="Minimum penetration depth of the foot tip into the ground (for soft terrain simulation). Default to 0.0 m",
    )
    parser.add_argument(
        "--g-p-max",
        type=float,
        default=0.02,
        help="Maximum penetration depth of the foot tip into the ground. Default to 0.02 m",
    )
    parser.add_argument(
        "--mp-tau",
        type=float,
        default=0.25,
        help="ime constant for low-pass filtering changes to morphological parameters, ensuring smooth transitions. Default to 0.25 s",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
        help="omputing device for simulation/training. Use `cuda` for GPU acceleration. Defaults to 'cpu'",
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=10.0,
        help="Duration of the CPG simulation in seconds. Default to 10.0 s",
    )
    parser.add_argument(
        "--enable-random-modulation",
        action="store_true",
        help="Adds random variation to CPG parameters for robustness testing. Disabled by default",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Output filename for saving simulation plots/images",
    )

    return parser.parse_args()


def make_command(enable_random_modulation: bool, net_size: int, device: str):
    if enable_random_modulation:
        return (
            torch.rand((1, net_size), device=device) * 2 - 1,
            torch.rand((1, net_size), device=device) * 2 - 1,
            torch.rand((1, 1), device=device) * 2 - 1,
        )

    return (
        torch.zeros((1, net_size), device=device),
        torch.zeros((1, net_size), device=device),
        torch.zeros((1, 1), device=device),
    )


def plot_hist(
    simualtion_time: float,
    X_hist: torch.Tensor,
    q1_hist: torch.Tensor,
    q1_max: float,
    Y_hist: torch.Tensor,
    q2_hist: torch.Tensor,
    q2_max: float,
    Z_hist: torch.Tensor,
    q3_hist: torch.Tensor,
    q3_max: float,
    s: float,
    h: float,
    d: float,
    g_c: float,
    g_p: float,
    w_max: float,
    omega: float,
    enable_random_modulation: bool,
    filename: str,
):
    x_axis = torch.linspace(0, simualtion_time, X_hist.shape[0])
    num_osc = X_hist.shape[1]

    mosaic = (("X", "q_1"), ("Y", "q_2"), ("Z", "q_3"))
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(20, 10), layout="constrained")

    var_dict = {
        "X": (
            f"Foot tip X coordinate (s: {s:.6f}, d: {d:.6f} m)",
            "$X$",
            X_hist,
            axes["X"],
        ),
        "q_1": (
            f"Joint 1 angle (max velocity: {q1_max:.6f} rad/s)",
            "$q_{1}$",
            q1_hist,
            axes["q_1"],
        ),
        "Y": (
            f"Foot tip Y coordinate (s: {s:.6f}, d: {d:.6f} m)",
            "$Y$",
            Y_hist,
            axes["Y"],
        ),
        "q_2": (
            f"Joint 2 angle (max velocity: {q2_max:.6f} rad/s)",
            "$q_{2}$",
            q2_hist,
            axes["q_2"],
        ),
        "Z": (
            f"Foot tip Z coordinate (h: {h:.6f} m, g_c: {g_c:.6f} m, g_p: {g_p:.6f} m)",
            "$Z$",
            Z_hist,
            axes["Z"],
        ),
        "q_3": (
            f"Joint 3 angle (max velocity: {q3_max:.6f} rad/s)",
            "$q_{3}$",
            q3_hist,
            axes["q_3"],
        ),
    }

    for var_name, (title, label, data, ax) in var_dict.items():
        for i in range(num_osc):
            ax.plot(x_axis, data[:, i], label=f"{label}[{i}]")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True)

    fig.suptitle(
        f"PF controller simulation (modulation: {enable_random_modulation}, omega: {omega / math.pi:.6f}$\\pi$ rad, w_max: {w_max / math.pi:.6f}$\\pi$ rad/s)",
        fontsize=16,
    )

    if filename:
        filepath = f"source/neurowalker/docs/images/{filename}.png"
        fig.savefig(filepath)
        print(f"\n[✓] Saving image to: {filepath}")

    plt.show()


def main():
    args_cli = parse_args()

    print("\n[✓] Starting simulation with the following parameters:\n")
    for key, value in vars(args_cli).items():
        print(f"{key:25} : {value}")

    cpg_cfg = HopfNetworkControllerCfg(
        dt=args_cli.dt,
        integration_method=args_cli.integration_method,
        a=args_cli.a,
        default_alpha=args_cli.default_alpha,
        mu_min=args_cli.mu_min,
        mu_max=args_cli.mu_max,
        w_min=args_cli.w_min,
        omega_cmd_min=args_cli.omega_cmd_min,
        omega_cmd_max=args_cli.omega_cmd_max,
        omega_cmd_tau=args_cli.omega_cmd_tau,
        coupling_cfg={
            "self_weight": args_cli.self_weight,
            "in_group_weight": args_cli.in_group_weight,
            "of_group_weight": args_cli.of_group_weight,
            "threshold": args_cli.threshold,
        },
    )
    cpg_controller = HopfNetworkController(
        cfg=cpg_cfg, num_envs=1, device=args_cli.device
    )

    ik_cfg = InverseKinematicsControllerCfg(
        dt=cpg_cfg.dt,
        s_min=args_cli.s_min,
        s_max=args_cli.s_max,
        d_min=args_cli.d_min,
        d_max=args_cli.d_max,
        h_min=args_cli.h_min,
        h_max=args_cli.h_max,
        g_c_min=args_cli.g_c_min,
        g_c_max=args_cli.g_c_max,
        g_p_min=args_cli.g_p_min,
        g_p_max=args_cli.g_p_max,
        mp_tau=args_cli.mp_tau,
    )
    ik_controller = InverseKinematicsController(
        cfg=ik_cfg,
        num_envs=cpg_controller.num_envs,
        device=cpg_controller.device,
    )

    w_max = (
        torch.ones((cpg_controller.num_envs, 1), device=cpg_controller.device)
        * args_cli.w_max
    )
    num_iterations = int(args_cli.simulation_time / cpg_cfg.dt)

    X_hist = torch.empty((num_iterations, cpg_controller.net_size), device="cpu")
    Y_hist = torch.empty_like(X_hist)
    Z_hist = torch.empty_like(X_hist)

    q1_hist = torch.empty((num_iterations, cpg_controller.net_size), device="cpu")
    q2_hist = torch.empty_like(q1_hist)
    q3_hist = torch.empty_like(q1_hist)

    if args_cli.enable_random_modulation:
        s_cmd = (
            torch.rand((1, 1), device=cpg_controller.device) * args_cli.s_max
            + args_cli.s_min
        )
        h_cmd = (
            torch.rand((1, 1), device=cpg_controller.device) * args_cli.h_max
            + args_cli.h_min
        )
        d_cmd = (
            torch.rand((1, 1), device=cpg_controller.device) * args_cli.d_max
            + args_cli.d_min
        )
        g_c_cmd = (
            torch.rand((1, 1), device=cpg_controller.device) * args_cli.g_c_max
            + args_cli.g_c_min
        )
        g_p_cmd = (
            torch.rand((1, 1), device=cpg_controller.device) * args_cli.g_p_max
            + args_cli.g_p_min
        )
    else:
        s_cmd = ik_controller.s
        h_cmd = ik_controller.h
        d_cmd = torch.zeros((1, 1), device=cpg_controller.device)
        g_c_cmd = torch.zeros((1, 1), device=cpg_controller.device)
        g_p_cmd = torch.zeros((1, 1), device=cpg_controller.device)

    exec_time = 0
    for i in range(num_iterations):
        mu, w, omega_cmd = make_command(
            args_cli.enable_random_modulation,
            cpg_controller.net_size,
            cpg_controller.device,
        )

        r, phi, omega = (
            cpg_controller.r.squeeze(0),
            cpg_controller.phi.squeeze(0),
            cpg_controller.omega.squeeze(0),
        )

        start_time = time.time_ns()
        cpg_controller.step(mu, w, w_max, omega_cmd)
        X_hist[i], Y_hist[i], Z_hist[i], q1_hist[i], q2_hist[i], q3_hist[i] = (
            ik_controller.solve_position(
                r, phi, omega, s_cmd, h_cmd, d_cmd, g_c_cmd, g_p_cmd
            )
        )
        exec_time += (time.time_ns() - start_time) / 1e6

    exec_time /= num_iterations

    print(
        f"\n[✓] Simulation ({num_iterations} steps, dt={cpg_cfg.dt}s) completed. Average controller step time: {exec_time:.3f} ms"
    )

    plot_hist(
        simualtion_time=args_cli.simulation_time,
        X_hist=X_hist,
        q1_hist=q1_hist,
        q1_max=q1_hist.diff(dim=0).max() / cpg_cfg.dt,
        Y_hist=Y_hist,
        q2_hist=q2_hist,
        q2_max=q2_hist.diff(dim=0).max() / cpg_cfg.dt,
        Z_hist=Z_hist,
        q3_hist=q3_hist,
        q3_max=q3_hist.diff(dim=0).max() / cpg_cfg.dt,
        s=ik_controller.s.item(),
        d=ik_controller.d.item(),
        h=ik_controller.h.item(),
        g_c=ik_controller.g_c.item(),
        g_p=ik_controller.g_p.item(),
        w_max=w_max.item(),
        omega=omega_cmd.item(),
        enable_random_modulation=args_cli.enable_random_modulation,
        filename=args_cli.filename,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
