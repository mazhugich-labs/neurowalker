"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
import argparse
import time

import torch
import matplotlib.pyplot as plt

from neurowalker.controllers.cpg import (
    HopfNetworkControllerCfg,
    HopfNetworkController,
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
        choices=("euler", "rk4"),
        default="euler",
        help="Numerical integration method for controller state estimation. Defaults to 'euler'",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=10,
        help="Convergence factor mean for the controller. Higher values make the system converge faster. Defaults to 10",
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

    return parser.parse_args()


def validate_bounds(mu_min: float, mu_max: float, w_min: float, w_max: float):
    if mu_min > mu_max:
        raise ValueError(
            f"mu_max cannot be lower than mu_min. Received mu_min: {mu_min}, mu_max: {mu_max}"
        )

    if w_min > w_max:
        raise ValueError(
            f"w_max cannot be lower than w_min. Received mu_min: {w_min}, mu_max: {w_max}"
        )


def make_command(enable_random_modulation: bool, net_size: int, device: str):
    if enable_random_modulation:
        return (
            torch.rand((1, net_size), device=device) * 2 - 1,
            torch.rand((1, net_size), device=device) * 2 - 1,
        )

    return (
        torch.zeros((1, net_size), device=device),
        torch.zeros((1, net_size), device=device),
    )


def plot_hist(
    simualtion_time: float,
    r_hist: torch.Tensor,
    d_r_hist: torch.Tensor,
    phi_hist: torch.Tensor,
    d_phi_hist: torch.Tensor,
    w_max: float,
    enable_random_modulation: bool,
):
    x_axis = torch.linspace(0, simualtion_time, r_hist.shape[0])
    num_osc = r_hist.shape[1]

    mosaic = (("r", "d_r"), ("phi", "d_phi"))
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(20, 10), layout="constrained")

    var_dict = {
        "r": ("Amplitude", "$r$", r_hist, axes["r"]),
        "d_r": ("Linear velocity", "$\\dot{r}$", d_r_hist, axes["d_r"]),
        "phi": ("Phase", "$\\phi$", phi_hist, axes["phi"]),
        "d_phi": ("Angular velocity", "$\\dot{\\phi}$", d_phi_hist, axes["d_phi"]),
    }

    for var_name, (title, label, data, ax) in var_dict.items():
        for i in range(num_osc):
            ax.plot(x_axis, data[:, i], label=f"{label}[{i}]")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True)

    fig.suptitle(
        f"CPG controller simulation (modulation: {enable_random_modulation}, w_max: {w_max / math.pi:.6f}$\\pi$ rad/s)",
        fontsize=16,
    )

    plt.show()


def main():
    args_cli = parse_args()
    validate_bounds(args_cli.mu_min, args_cli.mu_max, args_cli.w_min, args_cli.w_max)

    cfg = HopfNetworkControllerCfg(
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
    controller = HopfNetworkController(cfg, num_envs=1, device=args_cli.device)

    w_max = (
        torch.ones((controller.num_envs, 1), device=controller.device) * args_cli.w_max
    )
    num_iterations = int(args_cli.simulation_time / cfg.dt)

    r_hist = torch.empty((num_iterations, controller.net_size), device="cpu")
    d_r_hist = torch.empty_like(r_hist)
    phi_hist = torch.empty_like(r_hist)
    d_phi_hist = torch.empty_like(r_hist)

    exec_time = 0
    for i in range(num_iterations):
        mu, w = make_command(
            args_cli.enable_random_modulation,
            controller.net_size,
            controller.device,
        )

        r_hist[i] = controller.r.detach().squeeze(0).cpu()
        d_r_hist[i] = controller.d_r.detach().squeeze(0).cpu()
        phi_hist[i] = controller.phi.detach().squeeze(0).cpu()
        d_phi_hist[i] = controller.d_phi.detach().squeeze(0).cpu()

        start_time = time.time_ns()
        controller.step(mu, w, w_max)
        exec_time += (time.time_ns() - start_time) / 1e6

    exec_time /= num_iterations

    print(
        f"[✓] Simulation ({num_iterations} steps, dt={cfg.dt}s) completed. Average controller step time: {exec_time:.3f} ms"
    )

    plot_hist(
        args_cli.simulation_time,
        r_hist,
        d_r_hist,
        phi_hist,
        d_phi_hist,
        w_max.item(),
        args_cli.enable_random_modulation,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
