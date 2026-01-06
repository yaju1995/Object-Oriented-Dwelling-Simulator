import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from SRC.Controller.DDPGmodel.DDPG_Agent import DDPGAgent, DDPGConfig
from SRC.support import CustomLogger
from SRC.SIM.EquipmentClass import InverterModel, MeterModel, HVACModel, EVModel


logger = CustomLogger(command=True, color='green')


class ess_controller:
    def __init__(self, resolution:timedelta, control_period:timedelta= timedelta(minutes=30),agent:DDPGAgent =None):
        '''

        :param agent:
        :param resolution:
        :param control_period:
        '''
        self.agent = agent
        self. resolution = resolution
        self.control_period = control_period
        self.ess_controller_df = pd.DataFrame()

        # if self.agent = None in this condition use Rule based

    def update_status(self, meter_info:MeterModel, inverter_info:InverterModel):
        # Here I am get 1 min data need to convert that to
        # get state vs individual data
        # estimate load, pv, battery soc, tariffs

        #Update EV

        """

        if control_time >= control_period:
        evaluate the state over last period
        demand[next period], generation[next period], tariff[next period], feed_tariff[next period], soc[instant]

        Then
        Priority control
        EV, [15 min resolution ]
        HVAC, [5 min resolution ]
        ESS , [10 min resolution ] , minimum to support all the loads
        """
        pass

    def get_state(self, inverter_info: InverterModel, states_info = None):
        # based on the time we collect the state
        # 00.00.00 collect information instants wh with respect to control_period
        """
        :param time:
        :return:
        """
        '''
        Forecasted Load 
        Forecast generation 
        get next period tariff
        get next period feed tariff
        get instantaneous soc
        '''
        pass
