import pandas as pd
import pytz
import time
import numpy as np
import datetime as dt

from datetime import datetime
from loguru import logger
from itertools import cycle

from core.orchestrator import DataSourceConfig
from core.datacenter import DataCenterSrv
from core.algo_strat import AlgoStrategy
from strategy.strat_method import CreateSignal
from core.execution import SignalExecution


# start algo sequence
def algo_seq(BET_SIZE):
    start_time = dt.datetime.utcnow()
    logger.info('Starting algo_seq at (UTC) {} \n', start_time)

    # 1. Load strategy configuration
    ds = DataSourceConfig()
    ds.create_folder()
    strat_df = ds.load_info_dict()
    logger.info('Loaded strategy configuration with {} rows', len(strat_df))

    # 2. Build request / data frame
    dcs = DataCenterSrv(strat_df)
    dcs.create_df()
    logger.info('Do data cleaning and update data')

    # 3. Collect market data
    algo = AlgoStrategy(strat_df)
    algo.data_collect()
    logger.info('Data collection completed')

    # 4. Generate trading signals
    gen_signal = CreateSignal(strat_df)
    signal_df = gen_signal.split_sub()
    logger.info('Generated {} signals', len(signal_df))

    # 5. Execute signals with per-symbol bet sizes
    signal_exec = SignalExecution(signal_df, BET_SIZE)
    signal_exec.create_market_order()
    logger.info('Executed market orders with bet_size mapping: {}', BET_SIZE)

    end_time = dt.datetime.utcnow()
    logger.info('algo_seq finished at (UTC) {} (duration: {})', end_time, end_time - start_time)


# main to start
if __name__ == '__main__':
    BET_SIZE = {'BTC': 0.01}
    xx_min = {5, 15, 25, 35, 45, 55}

    algo_seq(BET_SIZE)