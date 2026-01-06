import pandas as pd
import numpy as np
from datetime import timedelta

from SRC.Controller.DDPGmodel.DDPG_Agent import DDPGAgent, DDPGConfig
from SRC.support.lib_config import CustomLogger
from SRC.SIM.EquipmentClass import EVModel
from SRC.Controller.Database.PandasDatabase import DataStore


from SRC.SIM.Tariff.tariffHandler import tariffHandler


logger = CustomLogger(command=True, color='green')


class ev_controller:
    def __init__(self, rl_agent: DDPGAgent, resolution:timedelta(minutes=1),update_period:timedelta= timedelta(minutes=30),
                 global_database:DataStore = None):
        self.global_database = global_database
        self.max_charging_power = 7
        self.connection_status = False
        self.connect_time = None
        self.leave_time = None
        self.full_charge_time = None
        self.connect_period = None
        self.resolution:timedelta = resolution
        self.update_period = update_period
        self.initial_soc = None
        self.final_soc = None
        self.full_charge_status = False
        ######################################################
        self.energy_state_info = None
        self.instant_state_info = None
        ######################################################
        self.rl_agent = rl_agent
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = False
        ######################################################
        self.update_time = None
        # self.now_time = None
        self.instant_power = 0
        self.instant_soc = 0
        self.period_charging_energy = 0
        self.period_charging_cost = 0
        self.ev_sessions_charging_energy = 0
        self.ev_sessions_charging_cost = 0
        self.ev_df = pd.DataFrame()
        self.set_charging_power = 1.5 # charge with minimal power
        self.ev_sessions = 0
        ######################################################
        self.tariff_handler = tariffHandler()
        self.data_storage = None
        ######################################################
        self.plotter = None

    def update_status(self, ev_info: EVModel):

        # --- Update EV dataframe ---
        self.ev_df = pd.concat(
            [self.ev_df, pd.DataFrame([ev_info.model_dump()]).set_index("time")]
        )

        tariff, feed_tariff = self.tariff_handler.get_tariff(ev_info.time)
        # ===============================================================
        #   HANDLE CONNECTION / DISCONNECTION EVENTS
        # ===============================================================
        if ev_info.ev_status:  # EV is connected
            if not self.connection_status:  # just now connected
                logger.commandline(f"Car connected {ev_info.time}")
                self.connection_status = True
                self.ev_sessions += 1
                self.connect_time = ev_info.time
                self.update_time = None  # reset control timer
                self.period_charging_energy = 0
                self.period_charging_cost = 0
                self.ev_sessions_charging_energy = 0
                self.ev_sessions_charging_cost = 0
                self.initial_soc = ev_info.ev_soc

        else:  # EV is not connected
            if self.connection_status:  # just disconnected
                # Print disconnection header
                logger.commandline(f"Car disconnected at {ev_info.time}")
                logger.commandline(f"Final reading → Power: {ev_info.ev_power:.2f} kW, SOC: {ev_info.ev_soc:.1f}%")

                self.connection_status = False
                self.leave_time = ev_info.time
                self.final_soc = self.instant_soc

                # --- FORCE FINAL REWARD FOR LAST ACTION ---
                self.control_logic(ev_info.time, True)

                # --- PRINT SESSION SUMMARY ---
                session_duration = self.leave_time - self.connect_time
                session_minutes = int(session_duration.total_seconds() // 60)
                if self.full_charge_status:
                    full_charge_duration = self.full_charge_time - self.connect_time
                    full_charge_minutes = f'{int(full_charge_duration.total_seconds() // 60)} minutes'
                else:
                    full_charge_minutes = 'Not fully charged'

                summary = (
                    "\n===== EV Charging Session Summary =====\n"
                    f"Arrival time:    {self.connect_time}\n"
                    f"Departure time:  {self.leave_time}\n"
                    f"Session length:  {session_minutes} minutes\n"
                    f"Initial SOC:     {self.initial_soc * 100:.1f}%\n"
                    f"Final SOC:       {self.final_soc * 100:.1f}%\n"
                    f"Energy charged:  {self.ev_sessions_charging_energy:.3f} kWh\n"
                    f"Total cost:      £{self.ev_sessions_charging_cost:.3f}\n"
                    f"Time to full:    {full_charge_minutes}\n"
                    f"Data points:     {len(self.ev_df)}\n"
                    "=======================================\n"
                )

                logger.commandline(summary)

            return None # If EV not connected return None
        # ===============================================================
        #   EV IS CONNECTED → UPDATE METRICS
        # ===============================================================
        # Update instantaneous state
        self.instant_power = ev_info.ev_power
        self.instant_soc = ev_info.ev_soc
        self.connect_period = ev_info.time - self.connect_time

        if self.instant_soc >= 1:
            self.full_charge_status = True # if full charged complete a cycle
            self.full_charge_time = ev_info.time
            # logger.commandline('EV Full charged !')
        else:
            self.full_charge_status = False

        # --- Compute incremental energy PER STEP ---
        # Energy [kWh] = kW * (minutes / 60)
        step_energy = self.instant_power * (self.resolution.total_seconds()/ 3600)
        # logger.commandline(self.resolution, self.resolution.total_seconds()/ 3600)
        self.period_charging_energy += step_energy
        self.ev_sessions_charging_energy += step_energy

        # Cost ONLY for this step:
        self.period_charging_cost += step_energy * tariff
        self.ev_sessions_charging_cost += step_energy * tariff

        # ===============================================================
        #   CONTROL UPDATE CHECK
        # ===============================================================
        # to do update in 14-min
        # period_minutes = int(self.update_period.total_seconds() // 60)
        # do_ev_update = (ev_info.time.minute % period_minutes == period_minutes - 1)
        # Update only when time aligns with resolution (e.g., 15 min)
        now_time = ev_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds()//60) == 0)

        # based on time delta
        # First update always allowed
        # if self.update_time is None:
        #     do_update = True
        # else:
        #     update_period = ev_info.time - self.update_time
        #     do_update = ev_info.time - self.update_time

        if do_update:
            # Apply control logic
            self.update_time = ev_info.time
            logger.commandline(f'Updating control signal using RL: {ev_info.time} {do_update}')

            self.control_logic(next_time, False)  # Update charging power using controller

            self.set_charging_power = self.action * self.max_charging_power

            # Reset period accumulators
            self.period_charging_energy = 0
            self.period_charging_cost = 0

        return self.set_charging_power # return in kW

    def control_logic(self, control_time, done):  # ddpg logic
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
            self.next_state = self.get_state(control_time)

            reward = self.compute_reward()

            # Save transition (terminal or non-terminal)
            self.rl_agent.store_transition(self.state, self.action, reward, self.next_state, done)
            # logger.commandline('Training RL agent!!')
            # logger.commandline(self.state, self.action, reward, self.next_state, done)
            self.rl_agent.train()

        # Check is the charge is disconnected : if so the set action = 0
        if done:
            # reset action to 0
            self.state = None
            self.action = None
            self.reward = None
            self.next_state = None
            self.action = 0
            return
        # ==========================================================
        # Update state for next action
        # ==========================================================
        self.state = self.get_state(control_time)
        logger.commandline(f'Getting States {self.state}')

        # ==========================================================
        # Choose next action [update self.action]
        # ==========================================================

        self.action = self.rl_agent.choose_action(self.state, noise_std=0.1)
        logger.commandline(f'Getting action {self.action}')
        return

    def get_state(self,control_time):
        """
        Get info from storage
        control_time is the next time period (15.59 now time the control time 15.59 + timedelta(resolution)
        => control time = 14.00 for resolution = 1 min
        State for RL agent.
        Must be normalized and numeric.
        soc : -(1-soc) handle the weight of ev charge [safely mapping for better exploration ]
        surplus with ev : reward on savings [net cost of energy during ev charging]
        surplus over next hour?

        tariff: tariff over next hour ?

        """
        ##########################################
        # ##### using memory to get information
        # df15 = self.global_database.past_period_resampled(
        #             control_time, past_period=1,
        #             out_resolution=self.update_period,
        #             headers=["Consumption (kW)", "Generation (kW)"],
        #             agg="mean",
        #         )
        # if df15 is not None:
        #     logger.commandline(f'control time : {control_time}')
        #     logger.commandline(df15)
        # use energy state info and instant state info to get states and reward
        ##################################################################
        # Normalized SOC (0–1)
        soc = self.instant_soc
        # next period tariff
        tariff_states, feed_tariff_states = self.tariff_handler.get_tariff_range_df(control_time, period=3, resolution=self.resolution)
        # print(f'tariff: {tariff_states}\n'
        #       f'feed: {feed_tariff_states}')
        # Charging duration in minutes → normalize by 720 (12 hours)
        # time to full charge

        if self.connect_period is None:
            connect_minutes = 0
        else:
            connect_minutes = int(self.connect_period.total_seconds() // 60)

        connect_norm = connect_minutes / 1440  # ~0–1 given a 24 hrs

         # Additional possible inputs: tariff, time of day, remaining time estimate ...
        # Finalize the EV states
        # Time of day min
        # Weekday and weekend
        return np.array([soc, *tariff_states, connect_norm], dtype=float)

    def compute_reward(self):
        """
        Compute reward for the current step.
        Reward must be independent, reusable, and not change state.
        """

        # Time connected so far (minutes)
        connect_minutes = int(self.connect_period.total_seconds() // 60)

        # SOC reward (0–1)
        soc_term = self.instant_soc / 100.0

        # Normalize time penalty (0–1 roughly)
        time_penalty = 0.01 * connect_minutes

        # Charging cost penalty (incremental for the period)
        cost_penalty = self.period_charging_cost

        reward = soc_term - time_penalty - cost_penalty

        # use the plotter to plot the reward values
        return reward


