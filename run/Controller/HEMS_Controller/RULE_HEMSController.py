from run.Controller.HEMS_Controller.controller_HEMS import HEMSController
import random
from time import perf_counter


class RuleController(HEMSController):
    def __init__(self, name, resolution, tariff_info, train):
        super().__init__(name, resolution, tariff_info, train)

    def control_logic(self, *args, **kwargs):
        t0 = perf_counter()

        data = self.get_observation()
        t1 = perf_counter()

        if data['feed tariff'][0] > data['tariff'][0]:
            self.control_signals.Battery_P_Setpoint = 1000
        else:
            self.control_signals.Battery_P_Setpoint = -1000

        self.control_signals.EV_Max_Power = random.randint(-1000, 1000)
        self.control_signals.HVAC_Heating_Power = random.randint(-5000, 5000)
        t2 = perf_counter()

        # print(f"get_observation: {(t1 - t0) * 1000:.3f} ms")
        # print(f"random assignments: {(t2 - t1) * 1000:.3f} ms")
        # print(f"total control_logic: {(t2 - t0) * 1000:.3f} ms")

    def get_observation(self):
        data = self.controller_database.get_past_period_n_data(now_time=self.time, period=4, keys=['Instant Cost',
                                                                                                   'tariff',
                                                                                                   'feed tariff',
                                                                                                   'Total Electric Power (kW)',
                                                                                                   'Battery Set Power (W)',
                                                                                                   'Battery Electric Power (kW)'])

        return data

    def get_from_db(self):
        pass