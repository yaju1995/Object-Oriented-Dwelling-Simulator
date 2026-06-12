from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta

from SRC.SIM.Tariff.tariffHandler_V_2_numpy import tariffHandler
from SRC.SIM.EquipmentClass import InverterModel, EVModel, HVACModel, MeterModel
from SRC.Controller.Database.numpyDatabase import DataStore
from SRC.SIM.ControlSignalHandler import ControlSignal
from run.Controller.base.base_controller import BaseController


@dataclass
class SensorInfo:
    ev: EVModel
    inverter: InverterModel
    hvac: HVACModel
    meter: MeterModel


class HEMSController(BaseController):
    def __init__(self, name: str, resolution: timedelta, tariff_info: tariffHandler, train: bool = False,
                 update_period: timedelta = timedelta(minutes=15)):
        super().__init__(name)

        self.tariff_info = tariff_info
        self.resolution = resolution
        self.train = train
        self.update_period = update_period

        self.controller_database = DataStore(resolution=resolution)  # might not be available
        self.control_signals = ControlSignal()

        self.info: SensorInfo | None = None

        self.time = None

    def update(self, ev_info: EVModel, inverter_info: InverterModel, hvac_info: HVACModel, meter_info: MeterModel):
        self.info = SensorInfo(
            ev=ev_info,
            inverter=inverter_info,
            hvac=hvac_info,
            meter=meter_info,
        )

        self.time = meter_info.time

        now_time = meter_info.time

        consumption = round(meter_info.active_power - inverter_info.battery_power + inverter_info.pv_power, 3)
        hours = self.resolution.total_seconds() / 3600

        # Cost of consumed power
        if meter_info.active_power > 0:
            instant_cost = round(meter_info.active_power * hours * meter_info.tariff, 4)
        else:
            instant_cost = round(meter_info.active_power * hours * meter_info.feed_tariff, 4)

        # Storing information in HVAC
        row = {  # Base on Constants COLUMNS_KEYS
            'Consumption (kW)': consumption,
            'Consumption (kWh)': consumption * hours,
            'Generation (kW)': inverter_info.pv_power,
            'Generation (kWh)': inverter_info.pv_power * hours,
            'Total Electric Power (kW)': meter_info.active_power,
            'Total Electric Power (kWh)': meter_info.active_power * hours,

            'tariff': meter_info.tariff,
            'feed tariff': meter_info.feed_tariff,
            'Instant Cost': instant_cost,

            'Battery SOC (-)': inverter_info.battery_soc,
            'Battery Set Power (W)': self.control_signals.Battery_P_Setpoint or 0,
            'Battery Electric Power (kW)': inverter_info.battery_power,
            'Battery Electric Energy (kWh)': inverter_info.battery_power * hours,

            'EV Parked': ev_info.ev_status,
            'EV SOC (-)': ev_info.ev_soc,
            'EV Set Point (kW)': self.control_signals.EV_Max_Power or 0,
            'EV Electric Power (kW)': ev_info.ev_power,
            'EV Electric Energy (kWh)': ev_info.ev_power * hours,

            'Temperature - Indoor (C)': hvac_info.ti,
            'Heating Electric Power (kW)': hvac_info.hvac_power,
            'Heating Electric Energy (kWh)': hvac_info.hvac_power * hours,
        }

        # Database test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.controller_database.append(now_time, row)  # First update row then collect information
        next_time = self.time + self.resolution
        do_update = (next_time.minute % (self.update_period.total_seconds() // 60) == 0)
        '''Add controller signal'''
        if do_update:
            self.control_logic()

        control_signal = self.control_signals.generate_control_signal()

        return control_signal

    def load_models(self, path: str = None):
        pass

    def save_models(self, path: str = None):
        pass

    def reset(self):
        pass

    @abstractmethod
    def control_logic(self, *args, **kwargs):
        """
              Add control logic in child class.

              Child classes can use:
                self.info.ev
                self.info.inverter
                self.info.hvac
                self.info.meter
                or the database to get observation or states
              """
        pass

    @abstractmethod
    def get_observation(self, *args, **kwargs):
        """
                Add observation required for control.
                """
        pass
