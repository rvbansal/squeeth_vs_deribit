import pandas as pd
import numpy as np


STRATEGY_DATA_PATH = './strategy_dfs'
PROCESSED_PATH = './processed_squeeth_datasets'


def build_squeeth_strategy_df(data_dict, freq='1H'):
    for _, df in data_dict.items():
        df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)

    def _build_ts(event, output_col):
        df = data_dict[event]
        grouper = pd.Grouper(key='utc_timestamp', freq=freq)
        output_col = df.groupby(grouper)[output_col]
        return output_col
  
    price_data_series = {
        'osqth_eth': _build_ts('uniswap_oe', 'price_standard').last(),
        'eth_usdc': _build_ts('uniswap_eu', 'price_standard').last(),
        'new_norm_factor': _build_ts('squeeth_norm_factor', 'newNormFactor').last(),
        'old_norm_factor': _build_ts('squeeth_norm_factor', 'oldNormFactor').last()
    }
    quantity_data_series = {
        'oe_pool_volume_in_eth': _build_ts('uniswap_oe', 'amount0').apply(lambda x: x.abs().sum()),
        'oe_pool_net_osqth': _build_ts('uniswap_oe', 'amount1').sum(),
        'oe_pool_net_eth': _build_ts('uniswap_oe', 'amount0').sum(),
        'oe_pool_add_eth': _build_ts('uniswap_oe_mint', 'amount0').sum(),
        'oe_pool_add_osqth': _build_ts('uniswap_oe_mint', 'amount1').sum(),
        'oe_pool_remove_eth': _build_ts('uniswap_oe_burn', 'amount0').sum(),
        'oe_pool_remove_osqth': _build_ts('uniswap_oe_burn', 'amount1').sum(),
        'oe_pool_fee_remove_eth': _build_ts('uniswap_oe_collect', 'amount0').sum(),
        'oe_pool_fee_remove_osqth': _build_ts('uniswap_oe_collect', 'amount1').sum(),
        'eu_pool_volume_in_eth': _build_ts('uniswap_eu', 'amount1').apply(lambda x: x.abs().sum())
    }
    price_ts_df = pd.DataFrame(price_data_series)
    quantity_ts_df = pd.DataFrame(quantity_data_series)
    
    full_range = pd.date_range(
        min(price_ts_df.index.min(), quantity_ts_df.index.min()),
        max(price_ts_df.index.max(), quantity_ts_df.index.max()), freq=freq
    )
    price_ts_df = price_ts_df.reindex(full_range).fillna(method='ffill')
    quantity_ts_df = quantity_ts_df.reindex(full_range).fillna(0)
    ts_df = pd.concat([price_ts_df, quantity_ts_df], axis=1)

    ts_df['usdc_eth'] = 1. / ts_df['eth_usdc']
    ts_df['eth_osqth'] = 1. / ts_df['osqth_eth']
    ts_df['usdc_osqth'] = ts_df['usdc_eth']*ts_df['eth_osqth']
    return ts_df


if __name__ == "__main__":
    squeeth_data_dict = {
        'uniswap_eu': pd.read_csv(PROCESSED_PATH + '/eth_usdc_data.csv'),
        'uniswap_oe_burn': pd.read_csv(PROCESSED_PATH + '/osqth_eth_burn_data.csv'),
        'uniswap_oe_mint': pd.read_csv(PROCESSED_PATH + '/osqth_eth_mint_data.csv'),
        'uniswap_oe_collect': pd.read_csv(PROCESSED_PATH + '/osqth_eth_collect_data.csv'),
        'uniswap_oe': pd.read_csv(PROCESSED_PATH + '/osqth_eth_data.csv'),
        'squeeth_liquidate': pd.read_csv(PROCESSED_PATH + '/liquidate_data.csv'),
        'squeeth_mint_short': pd.read_csv(PROCESSED_PATH + '/mint_short_data.csv'),
        'squeeth_norm_factor': pd.read_csv(PROCESSED_PATH + '/norm_factor_data.csv'),
        'squeeth_open_vault': pd.read_csv(PROCESSED_PATH + '/open_vault_data.csv'),
        'squeeth_withdraw_collateral': pd.read_csv(PROCESSED_PATH + '/withdraw_collateral_data.csv'),
        'squeeth_burn_short': pd.read_csv(PROCESSED_PATH + '/burn_short_data.csv'),
        'squeeth_deposit_collateral': pd.read_csv(PROCESSED_PATH + '/deposit_collateral_data.csv')
    }
    
    squeeth_strategy_df = build_squeeth_strategy_df(squeeth_data_dict)
    squeeth_strategy_df.to_csv(STRATEGY_DATA_PATH + '/squeeth_strategy_df.csv')
