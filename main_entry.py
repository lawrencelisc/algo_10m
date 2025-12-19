import pytz
import time
import numpy as np
import datetime as dt

from datetime import datetime
from loguru import logger

from core.orchestrator import DataSourceConfig
from core.datacenter import DataCenterSrv
from core.algo_strat import AlgoStrategy
from strategy.strat_method import CreateSignal
from core.execution import SignalExecution


# start algo sequence
def algo_seq(BET_SIZE):
    start_time = dt.datetime.now(dt.UTC)
    logger.info('Starting algo_seq at (UTC) {} \n', start_time)

    # 1. Load strategy configuration
    ds = DataSourceConfig()
    ds.create_folder()
    strat_df = ds.load_info_dict()
    logger.info('Loaded strategy configuration with {} rows', len(strat_df))

    # 2. Build request / data frame
    dcs = DataCenterSrv(strat_df)
    dcs.create_df()
    logger.info('Do data cleaning and update data complete')

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

    end_time = dt.datetime.now(dt.UTC)
    logger.info('algo_seq finished at (UTC) {} (duration: {})', end_time, end_time - start_time)


def next_scheduler(xx_min):
    time.sleep(0.5)
    now_dt = dt.datetime.now(tz=pytz.UTC)

    xx_min_sorted = sorted(xx_min)
    current_minute = now_dt.minute

    next_minute = None
    for m in xx_min_sorted:
        if m > current_minute or (m == current_minute and now_dt.second == 0):
            next_minute = m
            break

    if next_minute is None:
        next_minute = xx_min_sorted[0]
        next_runtime = (now_dt + dt.timedelta(hours=1)).replace(
            minute=next_minute,
            second=0,
            microsecond=0,
        )
    else:
        if current_minute == next_minute and now_dt.second == 0:
            idx = xx_min_sorted.index(next_minute)
            if idx + 1 < len(xx_min_sorted):
                next_minute = xx_min_sorted[idx + 1]
                next_runtime = now_dt.replace(
                    minute=next_minute,
                    second=0,
                    microsecond=0,
                )
            else:
                next_minute = xx_min_sorted[0]
                next_runtime = (now_dt + dt.timedelta(hours=1)).replace(
                    minute=next_minute,
                    second=0,
                    microsecond=0,
                )
        else:
            next_runtime = now_dt.replace(
                minute=next_minute,
                second=0,
                microsecond=0,
            )

    logger.info('next runtime schedule: {} in UTC time', next_runtime)
    return next_runtime


def scheduler(xx_min, BET_SIZE):
    count: int = 1

    try:
        while True:
            utc_now_dt = dt.datetime.now(tz=pytz.UTC)
            utc_now_min: int = utc_now_dt.minute
            utc_now_sec: int = utc_now_dt.second

            if (utc_now_min in xx_min) and (utc_now_sec == 0):
                logger.info(
                    'Start algo, UTC time={}, count # {}',
                    datetime.now(tz=pytz.UTC),
                    count,
                )
                # Call your algo
                algo_seq(BET_SIZE)

                # Show next schedule
                next_scheduler(xx_min)
                count += 1
                time.sleep(1.2)

            time.sleep(0.5)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.exception('Exception in scheduler: {}', e)


# main to start
if __name__ == '__main__':
    BET_SIZE = {'BTC': 0.001, 'ETH': 0.01}
    xx_min = {1, 11, 21, 31, 41, 51}

    utc_now = dt.datetime.now(dt.UTC)
    dt_str = utc_now.strftime('%Y-%m-%d %H:%M:%S')
    try:
        logger.info('Starting unified scheduler + algo program at (UTC) {}', dt_str)
        scheduler(xx_min, BET_SIZE)
    except KeyboardInterrupt:
        logger.warning('KeyboardInterrupt received; program terminated.')
    except Exception as e:
        logger.exception('Fatal error in main program: {}', e)