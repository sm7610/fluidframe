import argparse
import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from environment_taylor_green import TaylorGreenEnvironment


def load_policy(filename: str) -> np.ndarray:
    policy = np.load(filename)
    return policy


def plot_policy(
    n_episodes: int,
    positions: np.ndarray,
    positions_naive: np.ndarray,
    plot_params: Dict[str, float],
):
    ax2 = plt.subplot(111)
    delta_border = np.pi / 4

    x_min = np.min([positions[:, 0, :], positions_naive[:, 0, :]]) - delta_border
    x_max = np.max([positions[:, 0, :], positions_naive[:, 0, :]]) + delta_border
    y_min = np.min([positions[:, 1, :], positions_naive[:, 1, :]]) - delta_border
    y_max = np.max([positions[:, 1, :], positions_naive[:, 1, :]]) + delta_border

    x = np.linspace(x_min, x_max, int(100 * (x_max - x_min)))
    y = np.linspace(y_min, y_max, int(100 * (y_max - y_min)))
    X, Y = np.meshgrid(x, y)

    c = ax2.pcolormesh(
        X,
        Y,
        np.cos(X) * np.cos(Y),
        cmap="coolwarm",
        shading="auto",
        alpha=0.3,
        rasterized=True,
    )
    plt.colorbar(c, ax=ax2, shrink=0.5, label="vorticity")

    for episode in range(n_episodes):
        plt.plot(
            positions[:, 0, episode],
            positions[:, 1, episode],
            color="xkcd:rich purple",
            alpha=0.2,
        )
        plt.plot(
            positions_naive[:, 0, episode],
            positions_naive[:, 1, episode],
            color="xkcd:medium grey",
            alpha=0.2,
        )
        plt.plot(
            positions[-1, 0, episode],
            positions[-1, 1, episode],
            "o",
            markersize=2,
            markeredgecolor="xkcd:rich purple",
            markerfacecolor="none",
            alpha=0.7,
            label="trained" if episode == 0 else "",
        )
        plt.plot(
            positions_naive[-1, 0, episode],
            positions_naive[-1, 1, episode],
            "o",
            markersize=2,
            markeredgecolor="xkcd:medium grey",
            markerfacecolor="none",
            alpha=0.7,
            label="naïve" if episode == 0 else "",
        )

    plt.gca().set_aspect("equal")
    plt.legend(bbox_to_anchor=(1.05, 0.05), loc="lower left")
    plt.title(rf"$\phi={plot_params["phi"]}, \psi={plot_params["psi"]}$")
    plt.tight_layout()
    plt.savefig(f"phi{plot_params["phi"]}_psi{plot_params["psi"]}.pdf", dpi=300)


def eval(
    policy: list,
    swimmer_speed: float,
    alignment_timescale: float,
    n_episodes: int,
    n_steps: int,
    logging: bool = True,
    make_plot: bool = False,
) -> None:
    rng = np.random.default_rng(seed=42)
    env = TaylorGreenEnvironment(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=42,
    )  # instantiate environment

    env_naive = TaylorGreenEnvironment(
        dt=0.01,
        swimmer_speed=swimmer_speed,
        alignment_timescale=alignment_timescale,
        seed=42,
    )  # instantiate environment for naïve swimmer

    print(f"The trained policy is {policy}.")
    total_episode_return = 0
    total_episode_return_naive = 0
    positions = np.zeros([n_steps, 2, n_episodes])
    positions_naive = np.zeros([n_steps, 2, n_episodes])

    for episode in range(n_episodes):
        episode_return = 0
        episode_return_naive = 0

        position_initial = np.array(
            [rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)]
        )
        orientation_initial = rng.uniform(0, 2 * np.pi)

        obs = env.reset(position_initial.copy(), orientation_initial)
        _ = env_naive.reset(position_initial.copy(), orientation_initial)

        for i in range(n_steps):

            action = policy[obs]
            next_obs, reward = env.step(action)
            next_obs_naive, reward_naive = env_naive.step(1)

            obs = next_obs
            obs_naive = next_obs_naive

            episode_return += reward
            episode_return_naive += reward_naive

            positions[i, :, episode] = env.swimmer_position
            positions_naive[i, :, episode] = env_naive.swimmer_position

        total_episode_return += episode_return
        total_episode_return_naive += episode_return_naive

        if logging:
            print(
                f"Episode {episode+1} \t return: \t{episode_return} \t naïve return: \t{episode_return_naive}"
            )

    if logging:
        print(
            f"The mean return over {n_episodes} episodes is {total_episode_return/n_episodes}."
        )
        print(
            f"The mean naive return over {n_episodes} episodes is {total_episode_return_naive/n_episodes}."
        )

    print(f"The gain is {total_episode_return/total_episode_return_naive-1}.")

    if make_plot:
        plot_params = {"phi": env.swimmer_speed, "psi": env.alignment_timescale}
        plot_policy(n_episodes, positions, positions_naive, plot_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=list,
        # default=[1, 2, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],   # phi = 0.1, psi = 0.3
        # default = [3, 2, 2, 2, 1, 1, 1, 1, 3, 0, 3, 0],  # for TaylorGreenDedalusEnvironment
        default=[2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],  # phi = 0.3, psi = 1.0
        # default = [2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],  # for TaylorGreenDedalusEnvironment
        # default=[1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 1, 0],   # phi = 1.0, psi = 1.0
        # default = [1, 2, 1, 2, 1, 1, 1, 0, 1, 0, 1, 0],  # for TaylorGreenDedalusEnvironment
    )
    parser.add_argument("--swimmer-speed", type=float, default=0.3)
    parser.add_argument("--alignment-timescale", type=float, default=1.0)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=10000)
    parser.add_argument("--make-plot", type=bool, default=True)
    args = parser.parse_args()

    eval(
        policy=args.policy,
        swimmer_speed=args.swimmer_speed,
        alignment_timescale=args.alignment_timescale,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        make_plot=args.make_plot,
    )
