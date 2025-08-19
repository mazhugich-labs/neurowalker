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
        "--init-state-alpha",
        type=float,
        default=(0, math.pi, math.pi, 0, 0, math.pi),
        nargs="+",
        help="Initial oscillator phase offsets. Defaults to (0, math.pi, math.pi, 0, 0, math.pi), e.g. tripod gait of a 6-legged robot",
    )
    parser.add_argument(
        "--mu-min",
        type=float,
        default=1.0,
        help="Minimum allowed amplitude modulation. Defaults to 1.0",
    )
    parser.add_argument(
        "--mu-max",
        type=float,
        default=3.0,
        help="Maximum allowed amplitude modulation. Defaults to 4.0",
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
        default=0.05,
        help="Time constant of heading command low-pass filter. Smooths sharp turn commands. Defaults to 0.5 s",
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
            torch.ones((1, 1), device=device),
        )

    return (
        torch.zeros((1, net_size), device=device),
        torch.zeros((1, net_size), device=device),
        torch.zeros((1, 1), device=device),
    )


def plot_hist(
    simualtion_time: float,
    r_hist: torch.Tensor,
    delta_r_hist: torch.Tensor,
    phi_hist: torch.Tensor,
    delta_phi_hist: torch.Tensor,
    omega_hist: torch.Tensor,
    omega_cmd_hist: torch.Tensor,
    delta_omega_hist: torch.Tensor,
    w_max: float,
    enable_random_modulation: bool,
    filename: str,
):
    x_axis = torch.linspace(0, simualtion_time, r_hist.shape[0])
    num_osc = r_hist.shape[1]

    mosaic = (("r", "delta_r"), ("phi", "delta_phi"), ("omega", "delta_omega"))
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(20, 10), layout="constrained")

    var_dict = {
        "r": ("Amplitude", "$r$", r_hist, axes["r"]),
        "delta_r": ("Velocity", "$\\dot{r}$", delta_r_hist, axes["delta_r"]),
        "phi": ("Phase", "$\\phi$", phi_hist, axes["phi"]),
        "delta_phi": ("Frequency", "$\\dot{\\phi}$", delta_phi_hist, axes["delta_phi"]),
        "omega": (
            "Heading",
            ("$\\omega$", "$\\omega_{cmd}$"),
            (omega_hist, omega_cmd_hist),
            axes["omega"],
        ),
        "delta_omega": (
            "Heading frequency",
            "$\\dot{\\omega}$",
            delta_omega_hist,
            axes["delta_omega"],
        ),
    }

    for var_name, (title, label, data, ax) in var_dict.items():
        if var_name == "omega":
            ax.plot(x_axis, data[0][:], label=f"{label[0]}")
            ax.plot(x_axis, data[1][:], label=f"{label[1]}")
        elif var_name == "delta_omega":
            ax.plot(x_axis, data[:], label=f"{label}")
        else:
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

    cfg = HopfNetworkControllerCfg(
        integration_method=args_cli.integration_method,
        a=args_cli.a,
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
    controller = HopfNetworkController(
        cfg=cfg,
        init_alpha=torch.tensor(((0, math.pi, math.pi, 0, 0, math.pi),)),
        dt=args_cli.dt,
        num_envs=1,
        device=args_cli.device,
    )

    w_max = (
        torch.ones((controller.num_envs, 1), device=controller.device) * args_cli.w_max
    )
    num_iterations = int(args_cli.simulation_time / controller.dt)

    r_hist = torch.empty((num_iterations, controller.net_size), device="cpu")
    delta_r_hist = torch.empty_like(r_hist)
    phi_hist = torch.empty_like(r_hist)
    delta_phi_hist = torch.empty_like(phi_hist)
    omega_hist = torch.empty((num_iterations, 1), device="cpu")
    delta_omega_hist = torch.empty_like(omega_hist)
    omega_cmd_hist = torch.empty_like(omega_hist)

    exec_time = 0
    for i in range(num_iterations):
        mu, w, omega_cmd = make_command(
            args_cli.enable_random_modulation,
            controller.net_size,
            controller.device,
        )

        r_hist[i] = controller.r.detach().squeeze(0).cpu()
        delta_r_hist[i] = controller.delta_r.detach().squeeze(0).cpu()
        phi_hist[i] = controller.phi.detach().squeeze(0).cpu()
        delta_phi_hist[i] = controller.delta_phi.detach().squeeze(0).cpu()
        omega_hist[i] = controller.omega.detach().squeeze(0).cpu()
        delta_omega_hist[i] = controller.delta_omega.detach().squeeze(0).cpu()
        omega_cmd_hist[i] = omega_cmd.detach().squeeze(0).cpu() * math.pi

        start_time = time.time_ns()
        controller.step(mu, w, w_max, omega_cmd)
        exec_time += (time.time_ns() - start_time) / 1e6

    exec_time /= num_iterations

    print(
        f"\n[✓] Simulation ({num_iterations} steps, dt={controller.dt}s) completed. Average controller step time: {exec_time:.3f} ms"
    )

    plot_hist(
        simualtion_time=args_cli.simulation_time,
        r_hist=r_hist,
        delta_r_hist=delta_r_hist,
        phi_hist=phi_hist,
        delta_phi_hist=delta_phi_hist,
        omega_hist=omega_hist,
        omega_cmd_hist=omega_cmd_hist,
        delta_omega_hist=delta_omega_hist,
        w_max=w_max.item(),
        enable_random_modulation=args_cli.enable_random_modulation,
        filename=args_cli.filename,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
