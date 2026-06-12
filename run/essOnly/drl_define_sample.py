from run.Controller.deep_rl.DDPG.DDPGAgent import DDPGConfig, DDPGAgent

ess_agent_config = DDPGConfig(
    name="ESS_Bounded_DDPG",
    obs_dim=4,
    action_dim=1,
    return_mode="n_step",
    n_step=1,
    batch_size=128,
    buffer_capacity=100_000,
    gamma=0.99,
    tau=0.005,
    actor_lr=1e-4,
    critic_lr=1e-4,
    noise_std=0.1,

)

ess_agent = DDPGAgent(ess_agent_config)
import numpy as np
action = ess_agent.get_action(state=np.array([0,0,0,0]))

print(action)




