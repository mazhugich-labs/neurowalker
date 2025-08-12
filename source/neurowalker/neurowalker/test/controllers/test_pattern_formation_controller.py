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
from neurowalker.controllers.pf import (
    PatternFormationControllerCfg,
    PatternFormationController,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example script with random-command flag"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Controller update rate in seconds. Controls the simulation timestep. Defaults to 0.02 s",
    )
    parser.add_argument(
        "--integration-method",
        type=str,
        choices=("heun", "rk4"),
        default="heun",
        help="Numerical integration method for controller state estimation. Defaults to 'euler'",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=32,
        help="Convergence factor mean for the controller. Higher values make the system converge faster. Defaults to 32",
    )
    parser.add_argument(
        "--default-alpha",
        type=float,
        default=(0, math.pi, math.pi, 0, 0, math.pi),
        nargs="+",
        help="Default phase (in radians) for each oscillator. Defaults to (0, math.pi, math.pi, 0, 0, math.pi), e.g. tripod gait of a 6-legged robot",
    )
    parser.add_argument(
        "--mu-min",
        type=float,
        default=1.0,
        help="Lower bound for amplitude modulation parameter. Defaults to 1.0",
    )
    parser.add_argument(
        "--mu-max",
        type=float,
        default=4.0,
        help="Upper bound for amplitude modulation parameter. Defaults to 4.0",
    )
    parser.add_argument(
        "--w-min",
        type=float,
        default=0.0,
        help="Lower bound for frequency modulation parameter. Defaults to 0.0",
    )
    parser.add_argument(
        "--w-max",
        type=float,
        default=1.6 * math.pi,
        help="Upper bound for frequency modulation parameter. In real scenarios, this is should be modulated externally. Defaults to 1.6 * math.pi",
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
        help="Minimal phase difference (in radians) to consider oscillators as belonging to the same group. Defaults to 0.0 rad",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to perform computations on (CPU or GPU). Defaults to 'cpu'",
    )
    parser.add_argument(
        "--simulation-time",
        type=float,
        default=10.0,
        help="Total simulation time in seconds. Default to 10.0 s",
    )
    parser.add_argument(
        "--enable-random-modulation",
        action="store_true",
        help="Enables random modulation applied to the controller when set. Disabled by default",
    )
    parser.add_argument(
        "--d-step-min",
        type=float,
        default=0.0,
        help="Minimal step size in meters. Default to 0.0 m",
    )
    parser.add_argument(
        "--d-step-max",
        type=float,
        default=0.07,
        help="Maximal step size in meters. Default to 0.07 m",
    )
    parser.add_argument(
        "-h-min",
        type=float,
        default=0.07,
        help="Minimal robot height in meters. Default to 0.07 m",
    )
    parser.add_argument(
        "-h-max",
        type=float,
        default=0.12,
        help="Maximal robot height in meters. Default to 0.12 m",
    )
    parser.add_argument(
        "--g-c-min",
        type=float,
        default=0.02,
        help="Minimal foot tip ground clearance in meters. Default to 0.02 m",
    )
    parser.add_argument(
        "--g-c-max",
        type=float,
        default=0.07,
        help="Maximal foot tip ground clearance in meters. Default to 0.07 m",
    )
    parser.add_argument(
        "--g-p-min",
        type=float,
        default=0.0,
        help="Minimal foot tip ground penetration in meters. Default to 0.0 m",
    )
    parser.add_argument(
        "--g-p-max",
        type=float,
        default=0.02,
        help="Maximal foot tip ground penetration in meters. Default to 0.02 m",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename to save image",
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
    Y_hist: torch.Tensor,
    Z_hist: torch.Tensor,
    d_step: float,
    h: float,
    g_c: float,
    g_p: float,
    w_max: float,
    enable_random_modulation: bool,
    filename: str,
):
    x_axis = torch.linspace(0, simualtion_time, X_hist.shape[0])
    num_osc = X_hist.shape[1]

    mosaic = (("X",), ("Y",), ("Z",))
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(20, 10), layout="constrained")

    var_dict = {
        "X": (f"Foot tip X coordinate (d_step: {d_step:.6f} m)", "$X$", X_hist, axes["X"]),
        "Y": (f"Foot tip Y coordinate (d_step: {d_step:.6f} m)", "$Y$", Y_hist, axes["Y"]),
        "Z": (
            f"Foot tip Z coordinate (h: {h:.6f} m, g_c: {g_c:.6f} m, g_p: {g_p:.6f} m)",
            "$Z$",
            Z_hist,
            axes["Z"],
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
        f"PF controller simulation (modulation: {enable_random_modulation}, w_max: {w_max / math.pi:.6f}$\\pi$ rad/s)",
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

    pf_cfg = PatternFormationControllerCfg(
        d_step_min=args_cli.d_step_min,
        d_step_max=args_cli.d_step_max,
        h_min=args_cli.h_min,
        h_max=args_cli.h_max,
        g_c_min=args_cli.g_c_min,
        g_c_max=args_cli.g_c_max,
        g_p_min=args_cli.g_p_min,
        g_p_max=args_cli.g_p_max,
    )
    pf_controller = PatternFormationController(
        cfg=pf_cfg,
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

    exec_time = 0
    for i in range(num_iterations):
        mu, w, omega_cmd = make_command(
            args_cli.enable_random_modulation,
            cpg_controller.net_size,
            cpg_controller.device,
        )

        r, phi, xi = (
            cpg_controller.r.squeeze(0),
            cpg_controller.phi.squeeze(0),
            cpg_controller.xi.squeeze(0),
        )

        start_time = time.time_ns()
        X_hist[i], Y_hist[i], Z_hist[i] = pf_controller.solve_desired_pose(r, phi, xi)
        exec_time += (time.time_ns() - start_time) / 1e6

        cpg_controller.step(mu, w, w_max, omega_cmd)

    exec_time /= num_iterations

    print(
        f"\n[✓] Simulation ({num_iterations} steps, dt={cpg_cfg.dt}s) completed. Average controller step time: {exec_time:.3f} ms"
    )

    plot_hist(
        simualtion_time=args_cli.simulation_time,
        X_hist=X_hist,
        Y_hist=Y_hist,
        Z_hist=Z_hist,
        d_step=pf_controller.d_step.item(),
        h=pf_controller.h.item(),
        g_c=pf_controller.g_c.item(),
        g_p=pf_controller.g_p.item(),
        w_max=w_max.item(),
        enable_random_modulation=args_cli.enable_random_modulation,
        filename=args_cli.filename,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
