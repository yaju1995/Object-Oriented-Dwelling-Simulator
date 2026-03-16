from SRC.Controller.DDPGmodel.bounded_DDPG_Agent_multistep import DDPGConfig

from torch import nn

ESS_DDPG_config = DDPGConfig()
ESS_DDPG_config.gamma = 0.9
ESS_DDPG_config.actor_lr = 1e-3
ESS_DDPG_config.critic_lr = 1e-3
ESS_DDPG_config.hidden = (256,)
ESS_DDPG_config.activation = (nn.ReLU, nn.Tanh)
ESS_DDPG_config.batch_size = 500
ESS_DDPG_config.a_max = 1.0
ESS_DDPG_config.a_min = -1.0
ESS_DDPG_config.n_step = 4
# ESS_DDPG_config.tau = 1
# ESS_DDPG_config.seed = 42
ESS_LOOK_AHEAD = ESS_DDPG_config.n_step  # If you update this try to match the input state with this
ESS_INPUT_DIM = 1 + ESS_LOOK_AHEAD*2  # try to match the observed state with delay steps
ESS_OUT_DIM = 1


# agent = DPGAgent(ESS_INPUT_DIM, ESS_OUT_DIM, 1, gamma=0.9, seed=0)
ESS_MODEL_DIR = f'Models/ESS/test_nstep_res_15_{ESS_LOOK_AHEAD}_diff_tariff_up_4/'
ESS_MODEL_NAME = f'states_{ESS_INPUT_DIM}_delay_{ESS_LOOK_AHEAD}_1000eps.pth'