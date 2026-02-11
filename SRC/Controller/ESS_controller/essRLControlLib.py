import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from SRC.support.lib_config import CustomLogger
from SRC.Controller.Database.PandasDatabase import DataStore
from SRC.SIM.EquipmentClass import InverterModel, MeterModel
from SRC.support.live_plotter import LivePlotter

from SRC.Controller.DDPGmodel.DDPG_Agent_multistep import DDPGAgent
from SRC.Controller.DDPGmodel.bounded_DDPG_Agent_multistep import Bound_DDPGAgent

logger = CustomLogger(command=False, color='cyan')

def soc_charge_limit(state):
    soc = state[0]  # SOC should be 0 state

    charge_limit = min(0.5, 1-soc)
    discharge_limit = min(0.5, soc-0.05)
    return np.array([-discharge_limit]), np.array([charge_limit])


class essController:
    def __init__(self, rl_agent: Bound_DDPGAgent, mode='Train', resolution: timedelta = timedelta(minutes=1),
                 update_period: timedelta = timedelta(minutes=60),
                 global_database: DataStore = None,
                 max_charging_kw=5,
                 max_discharging_kw=5,
                 look_ahead=1,
                 energy_normalizer = 15):
        '''

        :param rl_agent:
        :param mode:
        :param resolution:
        :param update_period:
        :param global_database:
        :param max_charging_kw:
        :param max_discharging_kw:
        :param look_ahead: look ahead period for n step return
        :param energy_normalizer: Battery max capacity
        '''
        #############################################################
        self.resolution: timedelta = resolution
        self.update_period = update_period
        self.global_databased = global_database
        self.look_ahead = look_ahead
        #############################################################
        self.max_charging_power = max_charging_kw
        self.max_discharging_power = max_discharging_kw
        self.ESS_charge = False
        self.set_battery_power = 0

        #############################################################
        self.tariff_handler = None
        #############################################################
        self.rl_agent = rl_agent
        self.agent_name = rl_agent.name
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = False
        self.mode = mode
        self.cumulative_reward = 0
        #############################################################
        self.energy_normalizer = energy_normalizer
        ############################################################
        self.enable_plotter = True
        self.live_plotter = LivePlotter(title='Reward',
                                       xlabel='episode',
                                       ylabel='reward')

    def __str__(self):
        return (
            "ESSController(\n"
            f"  agent='{self.agent_name}',\n"
            f"  mode='{self.mode}',\n"
            f"  resolution={self.resolution},\n"
            f"  update_period={self.update_period},\n"
            f"  max_charge_kw={self.max_charging_power},\n"
            f"  max_discharge_kw={self.max_discharging_power},\n"
            f"  look_ahead={self.look_ahead},\n"
            f"  energy_normalizer={self.energy_normalizer},\n"
            f"  ESS_charge={self.ESS_charge},\n"
            f"  set_battery_power={self.set_battery_power},\n"
            f"  cumulative_reward={self.cumulative_reward}\n"
            ")"
        )

    def update_status(self, meter_info: MeterModel, inverter_info: InverterModel):
        done = False
        consumption = round(meter_info.active_power -
                            inverter_info.battery_power +
                            inverter_info.pv_power, 3)

        # ===============================================================
        #   CONTROL UPDATE CHECK
        # ===============================================================
        # to do update in 14-min
        # Update only when time aligns with resolution (e.g., 15 min)
        now_time = meter_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)
        if next_time.time() == time(0, 0, 0):
            done = True
        if do_update:  # every 15 or 30 mins update
            self.control_logic(next_time, done) # next time period is the control time period
            if self.action:
                self.set_battery_power = round(float(self.action) * self.energy_normalizer,6)
            else:
                self.set_battery_power = self.action
            # states = self.get_state(now_time)
            # logger.commandline(states)

        return self.set_battery_power

    def control_logic(self, control_time: datetime, done):  # ddpg logic
        '''
        Upadate self.action,
        :param control_time:
        :param now_time:
        :param done: if True then self.action = 0 no charging
        Else get action using rl_agent and update,
        :return: NA
        '''
        # ==========================================================
        # If we have a previous state-action pair, compute reward and update RL
        # ==========================================================
        if self.state is not None and self.action is not None:
            # Get reward
            reward = self.compute_reward(control_time)

            ## do 24 hhrs cumulative reward to check the optimization



            # print(reward, self.cumulative_reward)

            # Get next state #
            self.next_state = self.get_state(control_time)

            # Save transition (terminal or non-terminal)
            # pass to deque for delay reward update, update reward then add to agent store
            self.rl_agent.store_transition(self.state, self.action, reward, self.next_state, False)

            logger.commandline(control_time, self.state, self.action, reward, self.next_state, False)

            # Train agent
            # if update ready then only at the end of the session
            if self.mode == 'Train':
                if control_time.time() == time(0, 0, 0):
                    if self.enable_plotter:  # only plot after 24 hrs or something
                        self.live_plotter.update(self.cumulative_reward)
                    self.cumulative_reward = 0
                self.cumulative_reward += reward
                # update model every 24 hrs
                self.rl_agent.train()

        # Check is the charge is disconnected : if so the set action = 0
        if done:
            # compute reward and end
            # reset all state, reward, and action to None and set action to 0
            self.state = None
            self.action = None
            self.reward = None
            self.next_state = None
            # self.action = 0
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
        self.action = self.rl_agent.choose_action(self.state, noise_std=noise,bound_fn=soc_charge_limit)


        return

    def get_state(self, control_time: datetime):
        value = self.global_databased.get_instant_state(now_time=control_time, keys=['Consumption (kW)',
                                                                                     'Battery SOC (-)',
                                                                                   'Generation (kW)'])
        consumption = value.get('Consumption (kW)')/self.energy_normalizer
        soc = value.get('Battery SOC (-)')
        generation = value.get('Generation (kW)')/self.energy_normalizer

        surplus = generation-consumption
        # print(consumption, generation, self.energy_normalizer, surplus)
        tariff_df = self.tariff_handler.get_tariff_range_df(control_time, period=self.look_ahead,
                                                            resolution=self.update_period)
        tariff_states = tariff_df['tariff'].tolist()
        feed_tariff_states = tariff_df['feed_tariff'].tolist()
        tariff_states = np.array(tariff_states)
        tariff_max = self.tariff_handler.max_tariff
        tariff_min = self.tariff_handler.min_tariff
        tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min
        # print(value)
        # logger.commandline(f'state:{control_time}:{soc},{surplus},{tariff_states},->{tariff}')
        # pass
        return np.array([soc,surplus, *tariff ], dtype=float)

    def compute_reward(self, control_time: datetime):  # replace control time wiht self time??

        value = self.global_databased.get_instant_state(now_time=control_time, keys=['Instant Cost',
                                                                                     'tariff',
                                                                                     'Total Electric Power (kW)',
                                                                                     'Battery Set Power (W)',
                                                                                     'Battery Power (kW)'])

        # print(value)
        # Normalized tariff
        tariff_df = self.tariff_handler.get_tariff_range_df(control_time, period=1,
                                                            resolution=self.update_period)
        tariff_states = tariff_df['tariff'].tolist()
        feed_tariff_states = tariff_df['feed_tariff'].tolist()

        # tariff_states = np.array(tariff_states)
        # print(value)
        tariff_states = value.get('tariff')
        tariff_max = self.tariff_handler.max_tariff
        tariff_min = self.tariff_handler.min_tariff
        normalized_tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min
        # Normalized energy
        # delt_t = self.resolution.total_seconds() / 3600
        period_power = value.get('Total Electric Power (kW)')
        normalized_period_energy = period_power/self.energy_normalizer

        # Normalized reward
        cost = -(normalized_period_energy * normalized_tariff)
        # cost = value.get('Instant Cost')
        # logger.commandline(f'reward:{control_time}:{tariff_states},{normalized_tariff},{period_power},{normalized_period_energy},{cost}')
        # pass
        # check error is action does not match the reward
        set_power = value.get('Battery Set Power (W)') /1000
        actual_power = value.get('Battery Power (kW)')
        error_Reward = 0
        # print(set_power, actual_power)
        if round(set_power,3) - round(actual_power,3)> 0.001:
            print(f'Unbalance: {set_power}!={actual_power} ')
            logger.commandline(f'{set_power}!={actual_power} ')
            error_Reward = -5

        reward = cost + error_Reward

        # pass
        return reward

    def save_model(self, path):
        # Ensure the directory exists
        logger.commandline(self.rl_agent.save(path))

    def load_model(self, path):
        logger.commandline(self.rl_agent.load(path))
