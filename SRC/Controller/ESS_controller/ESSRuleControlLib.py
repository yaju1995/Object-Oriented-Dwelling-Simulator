import pandas as pd
import numpy as np
# from time import perf_counter

from datetime import timedelta
from SRC.support.lib_config import CustomLogger

from SRC.SIM.EquipmentClass import InverterModel, MeterModel

logger = CustomLogger(command=False, color='red')


class essController:
    def __init__(self, resolution: timedelta = timedelta(minutes=1),
                 update_period: timedelta = timedelta(minutes=60),
                 max_charging_kw=7,
                 max_discharging_kw=7,
                 look_ahead=1):
        self.resolution: timedelta = resolution
        self.update_period = update_period
        self.look_ahead = look_ahead
        #############################################################
        self.max_charging_power = max_charging_kw
        self.max_discharging_power = max_discharging_kw
        self.ESS_charge = False
        self.set_battery_power = 0

        #############################################################
        self.tariff_handler = None
        #############################################################
        self.total_ess_charging_cost = 0
        self.total_ess_charging_energy = 0

        self.total_ess_discharging_cost = 0
        self.total_ess_discharging_energy = 0

    def update_status(self, meter_info: MeterModel, inverter_info: InverterModel):

        # ===============================================================
        #   CONTROL UPDATE CHECK
        # ===============================================================
        # to do update in 14-min
        # Update only when time aligns with resolution (e.g., 15 min)
        now_time = meter_info.time
        next_time = now_time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)

        instant_power = inverter_info.battery_power
        step_energy = round(instant_power * (self.resolution.total_seconds() / 3600), 6)
        tariff, feed_tariff = self.tariff_handler.get_tariff(meter_info.time)
        if step_energy > 0:
            self.total_ess_charging_energy += abs(step_energy)
            self.total_ess_charging_cost += round(abs(step_energy) * tariff, 6)
        else:
            self.total_ess_discharging_energy += abs(step_energy)
            self.total_ess_discharging_cost += round(abs(step_energy) * tariff, 6)

        if do_update:  # every 15 or 30 mins update

            consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
            generation = round(inverter_info.pv_power, 3)
            surplus = generation - consumption
            next_tariff, next_feed_tariff = self.tariff_handler.get_tariff(next_time)
            ##############
            # if (inverter_info.battery_soc >= 0.05 and
            #         (surplus < 0 or
            #          next_feed_tariff < next_tariff)):
            #     # support load
            #     self.set_battery_power = surplus
            # elif (inverter_info.battery_soc < 100 and
            #       (surplus > 0 or
            #        next_feed_tariff > next_tariff)):
            #     # charge battery with surplus
            #     self.set_battery_power = surplus
            #     if next_feed_tariff > next_tariff: # change feed to average price self.tariff_handler.avg_tariff
            #         self.set_battery_power = self.max_charging_power
            # else:
            #     self.set_battery_power = 0
            #################
            if next_feed_tariff>next_tariff:
                self.set_battery_power = self.max_charging_power
            else:
                self.set_battery_power = surplus

        return self.set_battery_power

        #################################################################################################
        # if inverter_info.battery_soc <= 0.0:
        #     self.ESS_charge = True
        # elif inverter_info.battery_soc >= 0.8:
        #     self.ESS_charge = False
        #
        # if self.ESS_charge:
        #     # charge the battery at max
        #     self.set_battery_power = self.max_charging_power
        # else:
        #     # support the load in the house
        #     consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
        #     if (consumption / 2) > self.max_discharging_power:
        #         self.set_battery_power = -self.max_discharging_power
        #     else:
        #         self.set_battery_power = -(consumption / 2)
        # def getAction_Rule_based(self):
        #     # print('identify best course of action')
        #     energy_import = 0
        #     surplus = self.generation[self.time] - self.demand[self.time]
        #     if (self.storage.getStateOfCharge() > 20 and
        #             (surplus < 0 or self.export_price[self.time] < self.import_price[self.time])):
        #         # print('Discharge Energy')
        #         energy_import = -1 * self.demand[self.time]
        #
        #     elif (self.storage.getStateOfCharge() < 100 and
        #           (surplus > 0 or self.export_price[self.time] > self.import_price[self.time])):
        #         energy_import = 1 * self.storage.charging_power
        #
        #     # action = self.convertAction(energy_import)
        #     return energy_import


    def save(self):
        return f'No model to save!!!'

    def load(self):
        return f'No model to load!!!'
