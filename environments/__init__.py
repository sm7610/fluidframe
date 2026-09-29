from gymnasium.envs.registration import register

register(
    id="TaylorGreen-v0",
    entry_point="environments.taylor_green_gym:TaylorGreenContinuousGymEnv",
)
