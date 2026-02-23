import numpy as np
from SRC.Controller.ControlLib import controller
from datetime import timedelta, datetime, time
from SRC.SIM.EquipmentClass import InverterModel, EVModel, MeterModel, HVACModel

from SRC.SIM.EquipmentClass import InverterModel
from SRC.support.lib_config import CustomLogger
from SRC.support.live_plotter import LivePlotter, LivePlotter4
from SRC.Controller.DDPGmodel.bounded_DDPG_Agent_multistep import Bound_DDPGAgent
logger = CustomLogger(command=False, color='green')


class hvacController(controller):
    def __init__(self, resolution, global_database, update_period,
                 rl_agent:Bound_DDPGAgent,
                 mode = 'Train',
                 energy_normalizer = 1):
        super().__init__(resolution, global_database, update_period)
        self.set_HVAC_Power = None
        self.HVAC_ON = False
        #############################################################
        self.tariff_handler = None
        ############################################################
        self.rl_agent = rl_agent
        self.agent_name = rl_agent.name
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.mode = mode
        self.cumulative_reward = 0
        ############################################################
        self.energy_normalizer = energy_normalizer
        ############################################################
        self.enable_plotter = True
        self.no_sim = 0
        self.sum_reward = 0
        self.live_plotter = LivePlotter4(titles=['Reward', 'Avg Reward', 'Critic loss', 'Actor loss', ],
                                         xlabels=['Eps', 'Eps', 'Eps', 'Eps'],
                                         ylabels=['Reward', 'Avg Reward', 'Critic loss', 'Actor loss'])

    def __str__(self):
        return (
            "HVAC Controller("
            ")"
        )

    def update_status(self, hvac_info: HVACModel):
        now_time = hvac_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)
        if do_update:
            self.control_logic(next_time, False)
            if self.action is not None:
                self.set_HVAC_Power = round(float(self.action))
            else:
                self.set_HVAC_Power= self.action

        return self.set_HVAC_Power

    def control_logic(self, control_time: datetime, done):
        if self.state is not None and self.action is not None:
            # Get reward
            reward = self.compute_reward(control_time)
            self.cumulative_reward += reward

            # print(reward, self.cumulative_reward)

            # Get next state #
            self.next_state = self.get_state(control_time)

            # Save transition (terminal or non-terminal)
            # pass to deque for delay reward update, update reward then add to agent store
            self.rl_agent.store_transition(self.state, self.action, reward, self.next_state, False)
            # self.rl_agent.store_transition(self.state, self.action, reward, self.next_state)

            logger.commandline(control_time, self.state, self.action, reward, self.next_state, False)

            # Train agent
            # if update ready then only at the end of the session
            if self.mode == 'Train':
                # Update NN once a day
                if control_time.time() == time(0, 0, 0):
                    print(f'Updating Agent : {control_time}')
                    losses = self.rl_agent.train(batch_size=500)
                    if losses:
                        critic = losses.get('critic_loss')
                        actor = losses.get('actor_loss')
                    else:
                        critic = 0
                        actor = 0

                    if self.enable_plotter:  # only plot after 24 hrs or something
                        self.no_sim += 1
                        self.sum_reward += self.cumulative_reward
                        avg_reward = self.sum_reward / self.no_sim
                        self.live_plotter.update([self.cumulative_reward,
                                                  avg_reward,
                                                  critic,
                                                  actor,
                                                  ])

                    self.cumulative_reward = 0

                    # update model every 24 hrs

        # Check is the charge is disconnected : if so the set action = 0
        if done:
            # compute reward and end
            # reset all state, reward, and action to None and set action to 0
            self.state = None
            self.action = None
            self.reward = None
            self.next_state = None
            return
        # ==========================================================
        # Update state for next action
        # ==========================================================
        if self.next_state is not None:
            self.state = self.next_state
        else:
            self.state = self.get_state(control_time)

        # ==========================================================
        # Choose next action [update self.action]
        # ==========================================================
        if self.mode == 'Train':
            noise = 0.1
        else:
            noise = 0.0

        # safe layer implemented
        self.action = self.rl_agent.choose_action(self.state, noise_std=noise)

        return

    def get_state(self,control_time: datetime):
        value = self.global_databased.get_instant_state(now_time=control_time, keys=['Consumption (kW)',
                                                                                     'Battery SOC (-)',
                                                                                  'Generation (kW)'])

        consumption = value.get('Consumption (kW)') / self.energy_normalizer
        soc = value.get('Battery SOC (-)')
        generation = value.get('Generation (kW)') / self.energy_normalizer

        surplus = generation - consumption
        # print(consumption, generation, self.energy_normalizer, surplus)

        # getting next tariff
        tariff_df = self.tariff_handler.get_tariff_range_df(control_time, period=self.look_ahead,
                                                            resolution=self.update_period)
        tariff_states = tariff_df['tariff'].tolist()
        feed_tariff_states = tariff_df['feed_tariff'].tolist()
        tariff_states = np.array(tariff_states)
        im_tariff_max = self.tariff_handler.max_tariff
        im_tariff_min = self.tariff_handler.min_tariff

        feed_tariff_states = np.array(feed_tariff_states)
        exp_tariff_max = self.tariff_handler.max_feed_tariff
        exp_tariff_min = self.tariff_handler.min_feed_tariff

        tariff_max = max(exp_tariff_max, im_tariff_max)
        tariff_min = min(exp_tariff_min, im_tariff_min)

        tariff = np.round((tariff_states - tariff_min) / (tariff_max - tariff_min), 3)  # Normalized over max and min
        feed_tariff = np.round((feed_tariff_states - tariff_min) / (tariff_max - tariff_min), 3)
        # print(value)
        logger.commandline(f'state:{control_time}:{soc},{tariff_states},->{tariff},{feed_tariff_states}->{feed_tariff}')
        # pass
        return np.array([soc, *tariff, *feed_tariff], dtype=float)

    def compute_reward(self,control_time: datetime):
        value = self.global_databased.get_instant_state(now_time=control_time, keys=['Instant Cost',
                                                                                     'tariff',
                                                                                     'feed tariff',
                                                                                     'Total Electric Power (kW)',
                                                                                     'Battery Set Power (W)',
                                                                                     'Battery Power (kW)',
                                                                                     'HVAC Consumption (kW)',
                                                                                     'Temperature - Indoor (C)'])

        tariff_states = value.get('tariff')
        im_tariff_max = self.tariff_handler.max_tariff
        im_tariff_min = self.tariff_handler.min_tariff

        feed_tariff_states = value.get('feed tariff')
        feed_tariff_max = self.tariff_handler.max_feed_tariff
        feed_tariff_min = self.tariff_handler.min_feed_tariff

        tariff_max = max(feed_tariff_max, im_tariff_max)
        tariff_min = min(feed_tariff_min, im_tariff_min)

        normalized_tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min
        normalized_feed_tariff = (feed_tariff_states - tariff_min) / (
                    tariff_max - tariff_min)  # Normalized over max and min

        # Normalized energy
        hvac_period_power = value.get('HVAC Consumption (kW)')
        normalized_period_energy = hvac_period_power / self.energy_normalizer

        # Normalized reward
        if hvac_period_power >= 0:  # importing
            cost = -(normalized_period_energy * normalized_tariff)
        else:  # exporting
            cost = -(normalized_period_energy * normalized_feed_tariff)

        ########################################################################
        # thermal reward

        #
        return 0