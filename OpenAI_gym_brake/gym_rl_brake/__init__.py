from gym.envs.registration import register

register(
    id='rl_brake-v0',
    entry_point='gym_rl_brake.envs:SCADASEnv',
)