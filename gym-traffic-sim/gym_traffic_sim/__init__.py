from gym.envs.registration import register

register(
    id='traffic-sim-v0',
    entry_point='gym_traffic_sim.envs:TrafficSimEnv',
)