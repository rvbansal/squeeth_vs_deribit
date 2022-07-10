import pandas as pd
import numpy as np


STRATEGY_DATA_PATH = './strategy_dfs'
PROCESSED_PATH = './processed_deribit_datasets'


def symbol_freq_reindex(symbol_data, freq='1H'):
    date_range = pd.date_range(
        symbol_data['freq_timestamp'].min(), symbol_data['freq_timestamp'].max(), freq=freq
    )
    symbol_data = symbol_data.groupby('freq_timestamp').last()
    symbol_data = symbol_data.reindex(date_range).fillna(method='ffill')
    return symbol_data


def build_deribit_ts_df(concat_deribit_data, freq='1H'):
    for col in ['freq_timestamp', 'expiration', 'timestamp', 'local_timestamp']:
        concat_deribit_data[col] = pd.to_datetime(concat_deribit_data[col], utc=True)
        
    for col in ['mark_iv', 'bid_iv', 'ask_iv']:
        concat_deribit_data[col] /= 100
        
    symbol_reindexed_dfs = []
    for symbol, symbol_data in concat_deribit_data.groupby('symbol'):
        symbol_reindexed_df = symbol_freq_reindex(symbol_data, freq)
        symbol_reindexed_df['symbol'] = symbol
        symbol_reindexed_dfs.append(symbol_reindexed_df)
    reindexed_data = pd.concat(symbol_reindexed_dfs, axis=0)
    reindexed_data.index.name = 'freq_timestamp'
    
    reindexed_data['tte'] = (reindexed_data['expiration'] - reindexed_data['timestamp']).dt.days / 365.
    reindexed_data['opt_price_usdc'] = reindexed_data['mark_price']*reindexed_data['underlying_price']
    reindexed_data['opt_bid_price_usdc'] = reindexed_data['bid_price']*reindexed_data['underlying_price']
    reindexed_data['opt_ask_price_usdc'] = reindexed_data['ask_price']*reindexed_data['underlying_price']

    return reindexed_data


def reindex_puts_calls(calls_df, puts_df, columns = ['freq_timestamp', 'tte']):
    puts_df_temp = puts_df.reset_index().set_index(columns)
    calls_df_temp = calls_df.reset_index().set_index(columns)
    inters_index = puts_df_temp.index.intersection(calls_df_temp.index)
    puts_df = puts_df_temp.loc[inters_index]
    calls_df = calls_df_temp.loc[inters_index]
    puts_df = puts_df.reset_index().set_index('freq_timestamp')
    calls_df = calls_df.reset_index().set_index('freq_timestamp')
    return calls_df, puts_df


if __name__ == "__main__":
    call_deribit_data = pd.read_csv(PROCESSED_PATH + '/concat_call_options_chain_df.csv', index_col=0)
    put_deribit_data = pd.read_csv(PROCESSED_PATH + '/concat_put_options_chain_df.csv', index_col=0)

    call_ts_df = build_deribit_ts_df(call_deribit_data)
    put_ts_df = build_deribit_ts_df(put_deribit_data)
    call_ts_df, put_ts_df = reindex_puts_calls(call_ts_df, put_ts_df)
    
    concat_ts_df = pd.concat([call_ts_df, put_ts_df])
    concat_ts_df.to_csv(STRATEGY_DATA_PATH + '/deribit_strategy_df.csv')
