"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import math
from time import time_ns
import argparse

import torch
import matplotlib.pyplot as plt

from neurowalker.controllers.low_level import (
    HopfNetworkControllerCfg,
    HopfNetworkController,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example script with random-command flag"
    )
    parser.add_argument(
        "--random-command",
        action="store_true",
        help="Enable random command (default: False)",
    )
    return parser.parse_args()


def make_command(random_command, num_envs, net_size, device):
    if random_command:
        return (
            torch.rand((num_envs, net_size), device=device) * 2 - 1,
            torch.rand((num_envs, net_size), device=device) * 2 - 1,
        )

    return (
        torch.zeros((num_envs, net_size), device=device),
        torch.zeros((num_envs, net_size), device=device),
    )


def plot_history(
    x_axis, r_history, d_r_history, phi_history, d_phi_history, z_norm
):
    num_osc = r_history.shape[1]

    mosaic = (("r", "d_r"), ("phi", "d_phi"))
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 6), layout="constrained")

    var_dict = {
        "r": ("Amplitude $r$", r_history, axes["r"]),
        "d_r": ("Linear velocity $\\dot{r}$", d_r_history, axes["d_r"]),
        "phi": ("Phase $\\phi$ (rad)", phi_history, axes["phi"]),
        "d_phi": ("Angular velocity $\\dot{\\phi}$ (rad/s)", d_phi_history, axes["d_phi"]),
    }

    for var_name, (title, data, ax) in var_dict.items():
        for i in range(num_osc):
            ax.plot(x_axis, data[:, i], label=f"{var_name}[{i}]")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True)

    fig.suptitle(
        f"Low level controller simulation (z_norm: {z_norm})", fontsize=16
    )

    plt.show()


def main(total_time):
    args = parse_args()

    cfg = HopfNetworkControllerCfg(
        device="cuda",
        dt=0.02,
        a=10,
        omega=32 * math.pi,
        default_alpha=(
            0,
            2 * math.pi / 3,
            2 * math.pi / 3,
            4 * math.pi / 3,
            4 * math.pi / 3,
            0,
        ),
        mu_min=1.0,
        mu_max=4.0,
        w_min=0.0,
        z_norm_min=0.2,
        z_norm_max=1.0,
        coupling_cfg={
            "self_weight": 0.0,
            "in_group_weight": 1.0,
            "of_group_weight": 0.0,
            "threshold": 0.0,
        }
    )
    controller = HopfNetworkController(cfg, num_envs=1)

    num_iterations = int(total_time / cfg.dt)

    r_history = torch.empty((num_iterations, controller.net_size), device="cpu")
    d_r_history = torch.empty_like(r_history)
    phi_history = torch.empty_like(r_history)
    d_phi_history = torch.empty_like(r_history)

    if args.random_command:
        z_norm = torch.rand((controller.num_envs, 1), device=cfg.device)
    else:
        z_norm = torch.zeros((controller.num_envs, 1), device=cfg.device)

    start_time = time_ns()
    for i in range(num_iterations):
        mu, w = make_command(
            args.random_command, controller.num_envs, controller.net_size, cfg.device
        )

        r_history[i] = controller.r.squeeze(0).cpu()
        d_r_history[i] = controller.d_r.squeeze(0).cpu()
        phi_history[i] = controller.phi.squeeze(0).cpu()
        d_phi_history[i] = controller.d_phi.squeeze(0).cpu()

        controller.step(mu, w, z_norm)

    duration_ms = (time_ns() - start_time) / 1e6

    print(
        f"[✓] Simulation ({num_iterations} steps, dt={cfg.dt}s) completed in {duration_ms:.2f} ms"
    )

    x_axis = torch.linspace(0, total_time, steps=num_iterations)

    plot_history(
        x_axis,
        r_history,
        d_r_history,
        phi_history,
        d_phi_history,
        z_norm.item(),
    )


if __name__ == "__main__":
    main(total_time=10.0)
    simulation_app.close()
