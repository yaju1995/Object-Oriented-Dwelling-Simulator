from SRC.Controller.DDPGmodel.bounded_DDPG_Agent_multistep import DDPGConfig
from torch import nn

# DDPG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HVAC_DDPG_config = DDPGConfig()
# HVAC_DDPG_config.gamma = 0.9
# HVAC_DDPG_config.actor_lr = 1e-3
# HVAC_DDPG_config.critic_lr = 1e-3
# HVAC_DDPG_config.hidden = (128, 128)
# HVAC_DDPG_config.activation = (nn.ReLU, nn.Sigmoid)
# HVAC_DDPG_config.a_max = 1.0
# HVAC_DDPG_config.a_min = 0.0
# HVAC_LOOK_AHEAD = 4  # If you update this try to match the input state with this
# HVAC_INPUT_DIM = 4 + HVAC_LOOK_AHEAD  # try to match the observed state with delay steps
# HVAC_OUT_DIM = 1
# HVAC_MODEL_DIR = f'Models/HVAC/n_step_{HVAC_LOOK_AHEAD}/'
# HVAC_MODEL_NAME = f'states_{HVAC_INPUT_DIM}_config1_delay_{HVAC_LOOK_AHEAD}.pth'

# DQN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from SRC.Controller.DQNmodel.DQN_Agent import DQNConfig
HVAC_DQN_config  = DQNConfig()
HVAC_DQN_config.gamma = 0.9
HVAC_DQN_config.lr = 1e-4
HVAC_DQN_config.hidden = (256,128)
HVAC_DQN_config.seed = 0
HVAC_DQN_config.batch_size = 500
HVAC_LOOK_AHEAD = 1
HVAC_INPUT_DIM = 3 + HVAC_LOOK_AHEAD*2
HVAC_OUT_DIM = 2
HVAC_MODEL_DIR = f'Models/HVAC/test_n_step_{HVAC_LOOK_AHEAD}_bound_2_change_ref_2/'
HVAC_MODEL_NAME = f'DQN_states_{HVAC_INPUT_DIM}_config1_delay_{HVAC_LOOK_AHEAD}.pth'

action_map = {
    0: 0,
    1: 1
}