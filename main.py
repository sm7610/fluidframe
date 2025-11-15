import argparse

from environment_taylor_green import TaylorGreenEnvironment
from environment_taylor_green_dedalus import TaylorGreenDedalusEnvironment
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--swimmer-speed", type=float, default=0.3)
    parser.add_argument("--alignment-timescale", type=float, default=1.0)
    parser.add_argument("--n-episodes", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--use-dedalus-environment", action="store_true", default=False)
    args = parser.parse_args()

    if args.use_dedalus_environment:
        print("Using dedalus to specify flow variables ...")
        env = TaylorGreenDedalusEnvironment(
            dt=0.01,
            swimmer_speed=args.swimmer_speed,
            alignment_timescale=args.alignment_timescale,
            seed=42,
        )  # initialise environment
    else:
        print("Using closed-form analytical solution ...")
        env = TaylorGreenEnvironment(
            dt=0.01,
            swimmer_speed=args.swimmer_speed,
            alignment_timescale=args.alignment_timescale,
            seed=42,
        )  # initialise environment

    train(
        env=env,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        save=True,
        seed=42,
    )
