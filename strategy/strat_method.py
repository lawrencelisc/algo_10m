import pandas as pd
import numpy as np
import pytz
import sys
import os

from pathlib import Path
from loguru import logger
from datetime import datetime


class CreateSignal:
    # constant
    strat_list = ['zscore', 'ma_cross', 'bollinger', 'momentum', 'macd_quantile']
    strat_folder = Path('..') / 'data' / 'StratData'
    signal_folder = Path('..') / 'data' / 'Signal'
    signal_filename = 'signal_table.csv'
    signal_path = signal_folder / signal_filename


    def __init__(self, strat_df: pd.DataFrame):
        self.strat_df = strat_df
        print(self.strat_df)


    # gen signal table if strategy is zscore
    def strat_zscore(self, row):
        signal: int = 0
        name: str = str(row['name'])
        symbol: str = str(row['symbol'])
        endpt_col: str = str(row['endpt_col'])
        strat: str = str(row['strat'])
        mode: str = str(row['mode'])
        rol: int = int(row['rol'])
        thres: float = float(row['thres'])
        strat_filename: str = f'{name}_{endpt_col}_{symbol}.csv'
        file_path = self.strat_folder / strat_filename

        df = pd.read_csv(file_path, index_col=0)
        df[endpt_col] = df[endpt_col].astype('float64')
        df['ma'] = df[endpt_col].rolling(rol).mean().astype('float64')
        df['std'] = df[endpt_col].rolling(rol).std().astype('float64')
        df[strat] = ((df[endpt_col] - df['ma']) / df['std']).astype('float64')

        if mode == 'long':
            df['pos'] = np.select([df[strat] > thres], [1], default=0)
        elif mode == 'short':
            df['pos'] = np.select([df[strat] < thres], [-1], default=0)
        elif mode == 'long_short':
            df['pos'] = np.select([df[strat] > thres, df[strat] < thres], [1, -1], default=0)
        df.to_csv(file_path)
        signal = df['pos'].iloc[-1]

        return signal


    # gen signal table if strategy is ma cross
    def strat_ma_cross(self, row):
        signal: int = 0
        name: str = str(row['name'])
        symbol: str = str(row['symbol'])
        endpt_col: str = str(row['endpt_col'])
        strat: str = str(row['strat'])
        mode: str = str(row['mode'])
        short_rol: int = int(row['short_rol'])
        long_rol: int = int(row['long_rol'])
        strat_filename: str = f'{name}_{endpt_col}_{symbol}.csv'
        file_path = self.strat_folder / strat_filename

        df = pd.read_csv(file_path, index_col=0)
        df[endpt_col] = df[endpt_col].astype('float64')
        df[endpt_col] = df[endpt_col].replace('', np.nan)
        series = df[endpt_col].tolist()
        df['ewm_short'] = df[endpt_col].ewm(span=short_rol, min_periods=1).mean().astype('float64')
        df['sma_long'] = df[endpt_col].rolling(long_rol, min_periods=1).mean().astype('float64')

        if mode == 'long':
            df['pos'] = np.select([df['ewm_short'] > df['sma_long']], [1], default=0)
        elif mode == 'short':
            df['pos'] = np.select([df['ewm_short'] < df['sma_long']], [-1], default=0)
        elif mode == 'long_short':
            df['pos'] = np.select(
                [df['ewm_short'] > df['sma_long'], df['ewm_short'] < df['sma_long']],
                [1, -1],
                default=0
            )
        df.to_csv(file_path)
        signal = df['pos'].iloc[-1]

        return signal


    # gen signal table if strategy is bollinger
    def strat_bollinger(self, row):
        signal: int = 0
        name: str = str(row['name'])
        symbol: str = str(row['symbol'])
        endpt_col: str = str(row['endpt_col'])
        strat: str = str(row['strat'])
        mode: str = str(row['mode'])
        rol: int = int(row['rol'])
        num_std: float = float(row['num_std'])
        strat_filename: str = f'{name}_{endpt_col}_{symbol}.csv'
        file_path = self.strat_folder / strat_filename

        df = pd.read_csv(file_path, index_col=0)
        df[endpt_col] = df[endpt_col].astype('float64')
        df['ma'] = df[endpt_col].rolling(rol).mean().astype('float64')
        df['std'] = df[endpt_col].rolling(rol).std().astype('float64')
        df['std'] = np.where(df['std'] == 0, 1, df['std'])
        df['up'] = df['ma'] + (num_std * df['std'])
        df['dn'] = df['ma'] - (num_std * df['std'])

        if mode == 'long':
            df['pos'] = np.select(
                [df[endpt_col] > df['up'], df[endpt_col] < df['up']],
                [1, 0],
                default=0
            )
        elif mode == 'short':
            df['pos'] = np.select(
                [df[endpt_col] < df['dn'], df[endpt_col] > df['dn']],
                [-1, 0],
                default=0
            )
        elif mode == 'long_short':
            df['pos'] = np.select(
                [df[endpt_col] > df['up'], df[endpt_col] < df['dn']],
                [1, -1],
                default=0
            )
        df.to_csv(file_path)
        signal = df['pos'].iloc[-1]

        return signal


    # gen signal table if strategy is momentum
    def strat_momentum(self, row):
        signal: int = 0
        name: str = str(row['name'])
        symbol: str = str(row['symbol'])
        endpt_col: str = str(row['endpt_col'])
        strat: str = str(row['strat'])
        mode: str = str(row['mode'])
        rol: int = int(row['rol'])
        thres: float = float(row['thres'])
        strat_filename: str = f'{name}_{endpt_col}_{symbol}.csv'
        file_path = self.strat_folder / strat_filename

        df = pd.read_csv(file_path, index_col=0)
        df[endpt_col] = df[endpt_col].astype('float64')
        df['ma'] = df[endpt_col].rolling(rol).mean().astype('float64')
        df['ma_safe'] = np.where(df['ma'] == 0, 1, df['ma'])
        df['diff'] = ((df[endpt_col] - df['ma']) / df['ma_safe']) * 100

        if mode == 'long':
            df['pos'] = np.select([df['diff'] > thres], [1], default=0)
        elif mode == 'short':
            df['pos'] = np.select([df['diff'] < (-1 * thres)], [-1], default=0)
        elif mode == 'long_short':
            df['pos'] = np.select(
                [df['diff'] > thres, df['diff'] < (-1 * thres)],
                [1, -1],
                default=0
            )
        df.to_csv(file_path)
        signal = df['pos'].iloc[-1]

        return signal


    # gen signal table if strategy is macd quantile
    def strat_macd_quantile(self, row):

        # def ema(series: pd.Series, span: int):
        #    return series.ewm(span=span, adjust=False).mean()

        signal: int = 0
        fast: int = 12
        slow: int = 26
        signal_span: int = 9

        name: str = str(row['name'])
        symbol: str = str(row['symbol'])
        endpt_col: str = str(row['endpt_col'])
        strat: str = str(row['strat'])
        mode: str = str(row['mode'])
        rol: int = int(row['rol'])
        thres: float = float(row['thres'])
        strat_filename: str = f'{name}_{endpt_col}_{symbol}.csv'
        file_path = self.strat_folder / strat_filename

        df = pd.read_csv(file_path, index_col=0)
        df[endpt_col] = df[endpt_col].astype('float64')
        df['ema_fast'] = df[endpt_col].ewm(span=fast, adjust=False).mean()
        df['ema_slow'] = df[endpt_col].ewm(span=slow, adjust=False).mean()
        df['macd'] = df['ema_fast'] - df['ema_slow']
        lower_quan = thres * 0.01
        upper_quan = 1 - thres * 0.01
        df2 = df['macd'].rolling(window=rol, min_periods=rol)
        df['macd_upper'] = df2.quantile(upper_quan, interpolation='linear')
        df['macd_lower'] = df2.quantile(lower_quan, interpolation='linear')

        if mode == 'long':
            df['pos'] = np.select([df['macd'] > df['macd_upper']], [1], default=0)
        elif mode == 'short':
            df['pos'] = np.select([df['macd'] < df['macd_lower']], [-1], default=0)
        elif mode == 'long_short':
            df['pos'] = np.select([df['macd'] > df['macd_upper'],
                                   df['macd'] < df['macd_lower']],
                                  [1, -1],
                                  default=0
                                  )
        df.to_csv(file_path)
        signal = df['pos'].iloc[-1]

        return signal


    def split_sub(self):
        count = 0
        new_signal_df = pd.DataFrame()
        combine_signal_df = pd.DataFrame(
            {
                'name': pd.Series(dtype='string'),
                'symbol': pd.Series(dtype='string'),
                'saved_csv': pd.Series(dtype='string'),
                'signal': pd.Series(dtype='int64'),
            }
        )

        strategy_funcs = {
            self.strat_list[0]: self.strat_zscore,              # 0 - zscore
            self.strat_list[1]: self.strat_ma_cross,            # 1 - ma_cross
            self.strat_list[2]: self.strat_bollinger,           # 2 - bollinger
            self.strat_list[3]: self.strat_momentum,            # 3 - momentum
            self.strat_list[4]: self.strat_macd_quantile        # 4 - macd quantile
        }

        for _, row in self.strat_df.iterrows():
            func = strategy_funcs.get(row['strat'])
            if func:
                func(row)
                count += 1
                str_saved_csv = str(row['name'] + '_' + row['endpt_col'] + '_' + row['symbol'] + '.csv')
                strat_path = self.strat_folder / str_saved_csv
                df = pd.read_csv(strat_path, index_col=0)
                lastest_date = str(df.index[-1])

                new_row = {
                    'date': lastest_date,
                    'name': str(row['name']),
                    'symbol': str(row['symbol']),
                    'saved_csv': str_saved_csv,
                    'signal': str(func(row))
                }

                new_row_df = pd.DataFrame([new_row])
                new_row_df['date'] = pd.to_datetime(new_row_df['date'])
                new_row_df = new_row_df.set_index('date')

                frames = [df for df in [combine_signal_df, new_row_df] if df is not None and not df.empty]
                if frames:
                    combine_signal_df = pd.concat(frames, axis=0)
                else:
                    combine_signal_df = pd.DataFrame()

                # Try to load existing CSV if it exists
                if os.path.exists(self.signal_path):
                    try:
                        existing_signal_df = pd.read_csv(self.signal_path)
                    except pd.errors.EmptyDataError:
                        existing_signal_df = pd.DataFrame()
                        logger.error(f'Failed to read existing CSV {self.signal_filename}: {e}')
                else:
                    existing_signal_df = pd.DataFrame()

        # If the CSV is empty or missing, create it
        combine_signal_df = combine_signal_df.reset_index()
        if existing_signal_df.empty:
            combine_signal_df.to_csv(self.signal_path, index=False)
        else:
            combined_df = pd.concat([existing_signal_df, combine_signal_df], ignore_index=True)
            combined_df.to_csv(self.signal_path, index=False)
        logger.info(f'Updated {self.signal_filename}: +{count} new rows.')

        return combine_signal_df