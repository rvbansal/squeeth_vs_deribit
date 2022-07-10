import pandas as pd
import numpy as np


RAW_DOWNLOAD_PATH = './deribit_datasets'
PROCESSED_PATH = './processed_deribit_datasets'

START_DATE = '2022-01-10'
END_DATE = '2022-06-04'


def select_target_options(options_chain_df, asset='ETH', option_type='call'):
    options_chain_df = options_chain_df[options_chain_df['symbol'].str.startswith(asset)]
    options_chain_df = options_chain_df[options_chain_df['type'] == option_type]
    return options_chain_df


def time_format_options_chain_df(options_chain_df, freq='1H'):
    for col in ['timestamp', 'expiration', 'local_timestamp']:
        options_chain_df[col] = pd.to_datetime(options_chain_df[col], utc=True, unit='us')
    options_chain_df.sort_values('timestamp', inplace=True)
    options_chain_df['freq_timestamp'.format(freq)] = pd.DatetimeIndex(options_chain_df['timestamp']).\
        ceil(freq=freq)
    options_chain_df = options_chain_df.groupby(['symbol', 'freq_timestamp']).last()
    options_chain_df = options_chain_df.reset_index()
    return options_chain_df


def process_deribit_datasets(start_date=START_DATE, end_date=END_DATE):
    for date in pd.date_range(start_date, end_date, freq='D'):
        date_str = date.strftime('%Y-%m-%d')
        print('Processing options chain (Date = {}).'.format(date_str))

        raw_data = pd.read_csv(RAW_DOWNLOAD_PATH + '/deribit_options_chain_{}_OPTIONS.csv.gz'.format(date_str))
        options_chain_df = time_format_options_chain_df(raw_data)
        put_options_chain_df = select_target_options(options_chain_df, option_type='put')
        call_options_chain_df = select_target_options(options_chain_df, option_type='call')
        put_options_chain_df.to_csv(PROCESSED_PATH + '/put_options_chain_df_{}.csv'.format(date_str))
        call_options_chain_df.to_csv(PROCESSED_PATH + '/call_options_chain_df_{}.csv'.format(date_str))


def concat_processed_deribit_datasets(start_date=START_DATE, end_date=END_DATE, option_type='put'):
    output_dfs = []
    for date in pd.date_range(start_date, end_date, freq='D'):
        date_str = date.strftime('%Y-%m-%d')
        print('Reading {} options df (Date = {}).'.format(option_type, date_str))

        date_df = pd.read_csv(PROCESSED_PATH + '/{}_options_chain_df_{}.csv'.format(option_type, date_str))
        output_dfs.append(date_df)

    output_df = pd.concat(output_dfs)
    output_df.to_csv(PROCESSED_PATH + '/concat_{}_options_chain_df.csv'.format(option_type), index=False)


if __name__ == "__main__":
    process_deribit_datasets(START_DATE, END_DATE)
    concat_processed_deribit_datasets(START_DATE, END_DATE, option_type='call')
    concat_processed_deribit_datasets(START_DATE, END_DATE, option_type='put')

