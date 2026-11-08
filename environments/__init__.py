from gymnasium.envs.registration import register

register(
    id="StratifiedShear-v0",
    entry_point="environments.stratified_shear_gym:StratifiedShearGymEnv",
)

register(
    id="TaylorGreen-v0",
    entry_point="environments.taylor_green_gym:TaylorGreenContinuousGymEnv",
)
