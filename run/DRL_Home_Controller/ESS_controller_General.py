from abc import ABC

import numpy as np

from SRC.SIM.EquipmentClass import EVModel, InverterModel, HVACModel, MeterModel
from run.Controller.HEMS_Controller.DRL_HEMSController import DRLController


class ESSController(DRLController):
    def __init__(self, name, resolution, tariff_info, train, agent, look_ahead):
        super().__init__(name, resolution, tariff_info)

        self.look_ahead = look_ahead

    def get_observation(self) -> np.array | list | dict:  # only with instant observed information
        meter_info = self.info.meter
        inverter_info = self.info.inverter
        hvac_info = self.info.hvac
        ev_info = self.info.ev

        # next hour info
        next_time = self.time + self.resolution
        next_tariff, next_feed_tariff = self.tariff_info.get_tariff(next_time)
        # getting only power information
        return np.array([meter_info.tariff, next_tariff, next_feed_tariff, meter_info.feed_tariff,
                         meter_info.active_power,
                         ev_info.ev_soc, ev_info.ev_power, ev_info.ev_status,
                         hvac_info.hvac_power, hvac_info.ti, hvac_info.temp_ref])

    def get_reward(self) -> int:
        control_time = self.time

        value = self.controller_database.get_instant_state(now_time=control_time, keys=['Instant Cost',
                                                                                     'tariff',
                                                                                     'feed tariff',
                                                                                     'Total Electric Power (kW)',
                                                                                     'Battery Set Power (W)',
                                                                                     'Battery Electric Power (kW)'])

        # print(value)
        # tariff normalization
        tariff_states = value.get('tariff')
        im_tariff_max = self.tariff_info.max_tariff
        im_tariff_min = self.tariff_info.min_tariff

        feed_tariff_states = value.get('feed tariff')
        feed_tariff_max = self.tariff_info.max_feed_tariff
        feed_tariff_min = self.tariff_info.min_feed_tariff

        tariff_max = max(feed_tariff_max, im_tariff_max)
        tariff_min = min(feed_tariff_min, im_tariff_min)

        normalized_tariff = (tariff_states - tariff_min) / (tariff_max - tariff_min)  # Normalized over max and min
        normalized_feed_tariff = (feed_tariff_states - tariff_min) / (
                tariff_max - tariff_min)  # Normalized over max and min

        # Normalized energy
        period_power = value.get('Total Electric Power (kW)')
        period_energy = period_power * int(self.resolution.total_seconds() // 60) / 60
        normalized_period_energy = round(period_power / self.energy_normalizer, 6)

        # Normalized reward
        if period_power >= 0:  # importing
            normalized_cost = -round(normalized_period_energy * normalized_tariff, 6)
            period_cost = -round(period_energy * tariff_states, 6)
        else:  # exporting
            normalized_cost = -round(normalized_period_energy * normalized_feed_tariff, 6)
            period_cost = -round(period_energy * feed_tariff_states, 6)

        # check error is action does not match the reward
        set_power = value.get('Battery Set Power (W)') / 1000
        actual_power = value.get('Battery Electric Power (kW)')
        error_Reward = 0
        # print(set_power, actual_power)
        if round(set_power, 3) != round(actual_power, 3):
            print(f'Unbalance: {set_power}!={actual_power} ')
            error_Reward = -2

        reward = normalized_cost + error_Reward

        return reward

    def get_state(self) -> np.array:
        control_time = self.time
        forecasted_demand = self.info.inverter.forecast_demand
        forecasted_generation = self.info.inverter.forecast_generation

        value = self.controller_database.get_instant_state(now_time=control_time, keys=['Consumption (kW)',
                                                                                        'Battery SOC (-)',
                                                                                        'Generation (kW)'])

        consumption = value.get('Consumption (kW)')  # 11 kWH max data
        soc = value.get('Battery SOC (-)')
        generation = value.get('Generation (kW)')  # 5kW solar max

        surplus = round((generation - consumption) / 15, 6)
        forecast_surplus = round((forecasted_generation - forecasted_demand) / 15, 6)

        # getting next tariff
        tariff_df = self.tariff_info.get_tariff_range_df(control_time, period=self.look_ahead,
                                                         resolution=self.update_period)
        tariff_states = tariff_df['tariff'].tolist()
        feed_tariff_states = tariff_df['feed_tariff'].tolist()
        tariff_states = np.array(tariff_states)
        im_tariff_max = self.tariff_info.max_tariff
        im_tariff_min = self.tariff_info.min_tariff

        feed_tariff_states = np.array(feed_tariff_states)
        exp_tariff_max = self.tariff_info.max_feed_tariff
        exp_tariff_min = self.tariff_info.min_feed_tariff

        tariff_max = max(exp_tariff_max, im_tariff_max)
        tariff_min = min(exp_tariff_min, im_tariff_min)

        tariff = np.round((tariff_states - tariff_min) / (tariff_max - tariff_min), 3)  # Normalized over max and min
        feed_tariff = np.round((feed_tariff_states - tariff_min) / (tariff_max - tariff_min), 3)
        # print(value)
        print(
            f'state:{control_time}:::{soc}, {surplus}, {tariff_states},->{tariff},{feed_tariff_states}->{feed_tariff}')
        # pass
        return np.array([soc, forecast_surplus, *tariff, *feed_tariff], dtype=float)

    def save_models(self, path: str = None):
        pass

    def load_models(self, path: str = None):
        pass
