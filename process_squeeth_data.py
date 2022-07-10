from datetime import datetime
import json

import pandas as pd
from eth_abi import decode_abi
from web3 import Web3, constants


RAW_DOWNLOAD_PATH = './squeeth_datasets'
PROCESSED_PATH = './processed_squeeth_datasets'
ALCHEMY_KEY = ''

SQUEETH_SCALING = pow(10, -18)
UNISWAP_AMOUNT_SCALING = pow(10, -18)
ETH_USDC_PRICE_SCALING = pow(10, 12)


def get_event_call_inputs(contract_abi, event_call):
    for item in contract_abi:
        if 'name' in item.keys() and item['name'] == event_call and item['type'] == 'event':
            event_call_info = item
    
    event_call_inputs_unindexed = {}
    event_call_inputs_indexed = {}

    for item in event_call_info['inputs']:
        if item['indexed']:
            event_call_inputs_indexed[item['name']] = item['type']
        else:
            event_call_inputs_unindexed[item['name']] = item['type']
    return event_call_inputs_indexed, event_call_inputs_unindexed


def convert_json_to_df(event_name):
    data = json.load(open(RAW_DOWNLOAD_PATH + '/{}.json'.format(event_name)))
    event_info = json.load(open(RAW_DOWNLOAD_PATH + '/contract_abi/event_info.json'))

    contract_name = event_info[event_name]['contract_name']
    event_call = event_info[event_name]['event_call']
    contract_abi = json.load(open(RAW_DOWNLOAD_PATH + '/contract_abi/{}.json'.format(contract_name)))
    event_call_inputs_indexed,  event_call_inputs_unindexed = get_event_call_inputs(contract_abi, event_call)
    indexed_input_names = list(event_call_inputs_indexed.keys())
    indexed_input_types = list(event_call_inputs_indexed.values())
    unindexed_input_names = list(event_call_inputs_unindexed.keys())
    unindexed_input_types = list(event_call_inputs_unindexed.values())

    w3 = Web3(Web3.HTTPProvider(ALCHEMY_KEY))
    for data_item in data:
        data_item['data'] = decode_abi(unindexed_input_types, bytearray.fromhex(data_item['data'][2:]))

        data_item['event_signature'] = data_item['topics'][0]
        indexed_input_concat = ''.join([data_item['topics'][i + 1][2:] for i in range(len(indexed_input_types))])
        data_item['topics'] = decode_abi(indexed_input_types, bytearray.fromhex(indexed_input_concat))

        data_item['blockNumber'] = w3.toInt(hexstr=data_item['blockNumber'])
        data_item['logIndex'] = w3.toInt(hexstr=data_item['logIndex'])
        data_item['transactionIndex'] = w3.toInt(hexstr=data_item['transactionIndex'])

    df = pd.DataFrame(data)
    unindexed_input_cols = pd.DataFrame(df['data'].apply(pd.Series).values, columns=unindexed_input_names)
    indexed_input_cols = pd.DataFrame(df['topics'].apply(pd.Series).values, columns=indexed_input_names)
    df = pd.concat([df, unindexed_input_cols, indexed_input_cols], axis=1)
    df = df.sort_values(by = ['blockNumber', 'transactionIndex'])
    return df


def add_block_timestamps(df):
    block_ts_map = json.load(open(RAW_DOWNLOAD_PATH + '/block_timestamps.json'))
    df['unix_timestamp'] = df['blockNumber'].astype('string').map(block_ts_map)
    df['utc_timestamp'] = pd.to_datetime(df['unix_timestamp'], utc=True, unit='s')
    return df


def add_gas_info(df):
    w3 = Web3(Web3.HTTPProvider(ALCHEMY_KEY))

    def _get_gas_info(row):
        tx = w3.eth.get_transaction(row['transactionHash'])
        row['gas'] = tx.gas if 'gas' in tx else None
        row['gasPrice'] = tx.gasPrice if 'gasPrice' in tx else None
        row['maxFeePerGas'] = tx.maxFeePerGas if 'maxFeePerGas' in tx else None
        row['maxPriorityFeePerGas'] = tx.maxPriorityFeePerGas if 'maxPriorityFeePerGas' in tx else None
        return row
    
    df = df.apply(lambda row: _get_gas_info(row), axis=1)
    return df


def process_event_logs(name, gas_info, block_timestamps):
    df = convert_json_to_df(name)
    if block_timestamps: 
        df = add_block_timestamps(df)
    if gas_info: 
        df = add_gas_info(df)
    return df


def save_processed_data(df, name):
    file_name = PROCESSED_PATH + '/{}_data.csv'.format(name)
    print('Saved to {}.'.format(file_name))
    df.to_csv(file_name, index=False)


def standardize_uniswap_v3_sqrt_prices(prices):
    return (prices ** 2) / (2 ** 192)


def standardize_uniswap_v3_tick_to_prices(ticks):
    return 1.0001**ticks


def process_norm_factor_data(name = 'norm_factor', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['oldNormFactor'] *= SQUEETH_SCALING
    df['newNormFactor'] *= SQUEETH_SCALING
    df['lastModificationTimestamp'] = pd.to_datetime(df['lastModificationTimestamp'], utc=True, unit='s')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, unit='s')
    save_processed_data(df, name)


def process_burn_short_data(name = 'burn_short', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount'] *= SQUEETH_SCALING
    save_processed_data(df, name)


def process_mint_short_data(name = 'mint_short', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount'] *= SQUEETH_SCALING
    save_processed_data(df, name)


def process_deposit_collateral_data(name = 'deposit_collateral', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount'] *= SQUEETH_SCALING
    save_processed_data(df, name)


def process_withdraw_collateral_data(name = 'withdraw_collateral', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount'] *= SQUEETH_SCALING
    save_processed_data(df, name)


def process_deposit_uni_position_data(name = 'deposit_uni_position', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    save_processed_data(df, name)


def process_withdraw_uni_position_data(name = 'withdraw_uni_position', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    save_processed_data(df, name)


def process_liquidate_data(name = 'liquidate', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['debtAmount'] *= SQUEETH_SCALING
    df['collateralPaid'] *= SQUEETH_SCALING
    save_processed_data(df, name)


def process_open_vault_data(name = 'open_vault', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    save_processed_data(df, name)


def process_osqth_eth_data(name = 'osqth_eth', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['price_standard'] = standardize_uniswap_v3_sqrt_prices(df['sqrtPriceX96'])
    df['amount0'] *= UNISWAP_AMOUNT_SCALING
    df['amount1'] *= UNISWAP_AMOUNT_SCALING
    save_processed_data(df, name)


def process_osqth_eth_burn_data(name = 'osqth_eth_burn', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount0'] *= UNISWAP_AMOUNT_SCALING
    df['amount1'] *= UNISWAP_AMOUNT_SCALING
    df['amount'] *= UNISWAP_AMOUNT_SCALING
    df['tickLower_standard'] = standardize_uniswap_v3_tick_to_prices(df['tickLower'])
    df['tickUpper_standard'] = standardize_uniswap_v3_tick_to_prices(df['tickUpper'])
    save_processed_data(df, name)


def process_osqth_eth_mint_data(name = 'osqth_eth_mint', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount0'] *= UNISWAP_AMOUNT_SCALING
    df['amount1'] *= UNISWAP_AMOUNT_SCALING
    df['amount'] *= UNISWAP_AMOUNT_SCALING
    df['tickLower_standard'] = standardize_uniswap_v3_tick_to_prices(df['tickLower'])
    df['tickUpper_standard'] = standardize_uniswap_v3_tick_to_prices(df['tickUpper'])
    save_processed_data(df, name)


def process_osqth_eth_collect_data(name = 'osqth_eth_collect', gas_info=True, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['amount0'] *= UNISWAP_AMOUNT_SCALING
    df['amount1'] *= UNISWAP_AMOUNT_SCALING
    df['tickLower_standard'] = standardize_uniswap_v3_tick_to_prices(df['tickLower'])
    df['tickUpper_standard'] = standardize_uniswap_v3_tick_to_prices(df['tickUpper'])
    save_processed_data(df, name)


def process_eth_usdc_data(name = 'eth_usdc', gas_info=False, block_timestamps=True):
    df = process_event_logs(name, gas_info, block_timestamps)
    df['price_standard'] = standardize_uniswap_v3_sqrt_prices(df['sqrtPriceX96'])
    df['price_standard'] /= ETH_USDC_PRICE_SCALING
    df['amount0'] *= UNISWAP_AMOUNT_SCALING * ETH_USDC_PRICE_SCALING
    df['amount1'] *= UNISWAP_AMOUNT_SCALING
    save_processed_data(df, name)


if __name__ == "__main__":
    process_norm_factor_data()
    process_burn_short_data()
    process_mint_short_data()
    process_deposit_collateral_data()
    process_withdraw_collateral_data()
    process_deposit_uni_position_data()
    process_withdraw_uni_position_data()
    process_liquidate_data()
    process_open_vault_data()
    process_osqth_eth_data()
    process_osqth_eth_burn_data()
    process_osqth_eth_mint_data()
    process_osqth_eth_collect_data()
    process_eth_usdc_data()
