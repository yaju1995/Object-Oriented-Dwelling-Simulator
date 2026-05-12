# from SRC.Controller.DDPGmodel.bounded_DDPG_Agent_multistep import DDPGConfig
from SRC.Controller.DDPGmodel.DDPG_Agent_multistep import DDPGConfig
from torch import nn

EV_DDPG_config = DDPGConfig()
EV_DDPG_config.gamma = 0.9
EV_DDPG_config.actor_lr = 1e-3
EV_DDPG_config.critic_lr = 1e-3
EV_DDPG_config.hidden = (128, 128)
EV_DDPG_config.activation = (nn.ReLU, nn.Sigmoid)
EV_DDPG_config.a_max = 1.0
EV_DDPG_config.a_min = 0.0
EV_DDPG_config.seed = 3
# EV_DDPG_config.batch_size = 1000
EV_LOOK_AHEAD = 1  # If you update this try to match the input state with this
EV_INPUT_DIM = 3 + EV_LOOK_AHEAD  # try to match the observed state with delay steps
EV_OUT_DIM = 1
EV_MODEL_DIR = f'Models/EV/test_5_n_step_{EV_LOOK_AHEAD}_setup_1/Seed_{EV_DDPG_config.seed}/'
EV_MODEL_NAME = f'states_{EV_INPUT_DIM}_config1_delay_{EV_LOOK_AHEAD}_cost_4.pth'