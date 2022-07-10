import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


STRATEGY_DATA_PATH = './strategy_dfs'


def compute_instrument_prices(
    deribit_df, 
    max_spread_bid_ratio=0.12, 
    max_spread_width=0.6, 
    min_spread_width=0.05,
    min_instrument_premium=0.0001
):
    deribit_df['spread_width'] = deribit_df['ask_price'] - deribit_df['bid_price']
    deribit_df['prev_mark_price'] = deribit_df.groupby('symbol')['mark_price'].transform(lambda g: g.shift(1))

    max_spread_term = np.minimum(
        max_spread_bid_ratio*deribit_df['bid_price'], max_spread_width
    )
    too_wide = deribit_df['spread_width'] >= np.maximum(max_spread_term, min_spread_width)
    
    instrument_price = (deribit_df['ask_price'] + deribit_df['bid_price']) / 2
    instrument_price.loc[too_wide] = deribit_df.loc[too_wide, 'prev_mark_price']
    null_instrument_price = instrument_price.isnull()
    instrument_price.loc[null_instrument_price] = deribit_df.loc[null_instrument_price, 'mark_price']
    deribit_df['instrument_price'] = instrument_price

    deribit_df = deribit_df[deribit_df['instrument_price'] > min_instrument_premium]
    return deribit_df


def compute_forward_prices(deribit_df):
    
    def _get_forward_price(date_df):
        date_df = date_df.set_index('strike_price')
        calls = date_df.loc[date_df['type'] == 'call']
        puts = date_df.loc[date_df['type'] == 'put']
        call_prices = calls['instrument_price']
        put_prices = puts['instrument_price']

        if len(call_prices[~call_prices.isnull()]) < 2 or len(put_prices[~put_prices.isnull()]) < 2:
            return np.nan

        strike_min = np.abs(call_prices - put_prices).idxmin()
        if np.isnan(strike_min):
            return np.nan
        
        forward_price = strike_min / (1 - call_prices[strike_min] + put_prices[strike_min])
        return forward_price
    
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    forward_prices = grouped_df.apply(_get_forward_price).rename('forward_price')
    deribit_df = deribit_df.join(forward_prices, on=['freq_timestamp', 'tte']).reset_index()
    return deribit_df


def compute_strike_cutoffs(deribit_df):
    
    def _get_strike_cutoff(date_df):
        date_df = date_df[date_df['strike_price'] <= date_df['forward_price']]
        if len(date_df) == 0:
            return np.nan
        min_idx = (date_df['strike_price'] - date_df['forward_price']).abs().idxmin()
        strike_cutoff = date_df.loc[min_idx, 'strike_price']
        return strike_cutoff
    
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    strike_cutoffs = grouped_df.apply(_get_strike_cutoff).rename('strike_cutoff')
    deribit_df = deribit_df.join(strike_cutoffs, on=['freq_timestamp', 'tte']).reset_index()
    deribit_df = deribit_df.drop('index', axis=1)
    return deribit_df


def get_strike_width_and_prices(df, for_2perp=False):
    calls = df[df['type'] == 'call']
    puts = df[df['type'] == 'put']
    
    if len(calls) == 0 or len(puts) == 0:
        df['strike_width'] = np.nan
        df['avg_instrument_price'] = np.nan
        return df
    
    df = df.sort_values(by='strike_price')
    
    strikes = df['strike_price'].drop_duplicates().values
    bigger_strikes = df['strike_price'].shift(-1).values
    smaller_strikes = df['strike_price'].shift(1).values

    strike_widths = (bigger_strikes - smaller_strikes) / 2
    strike_widths_map = {s: w for s, w in zip(strikes, strike_widths)}
    strike_widths_map[strikes.min()] = np.nanmin(bigger_strikes) - strikes.min()
    strike_widths_map[strikes.max()] = strikes.max() - np.nanmax(smaller_strikes)
    if for_2perp:
        strike_widths_map[strikes.min()] += strikes.min()
        strike_widths_map[strikes.max()] += strikes.max()
    
    df['strike_width'] = df['strike_price'].map(strike_widths_map)

    combined_price_map = df.groupby('strike_price')['instrument_price'].mean().to_dict()
    df['avg_instrument_price'] = df['strike_price'].map(combined_price_map)
    return df


def filter_to_otm_options(deribit_df):
    otm_calls = (deribit_df['type'] == 'call') & (deribit_df['strike_price'] >= deribit_df['strike_cutoff'])
    otm_puts = (deribit_df['type'] == 'put') & (deribit_df['strike_price'] <= deribit_df['strike_cutoff'])
    deribit_df = deribit_df[otm_calls | otm_puts]
    return deribit_df


def compute_dvols(deribit_df):
    deribit_df = filter_to_otm_options(deribit_df)
        
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    deribit_df = grouped_df.apply(get_strike_width_and_prices).reset_index(drop=True)
    
    strike_cont_num = deribit_df['avg_instrument_price']*deribit_df['strike_width']*deribit_df['forward_price']
    strike_cont_den = deribit_df['strike_price']**2
    deribit_df['dvol_strike_cont'] = strike_cont_num / strike_cont_den
    
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    deribit_df['dvol_strike_cont_sum'] = grouped_df['dvol_strike_cont'].transform('sum')
    
    ivar_num = 2*deribit_df['dvol_strike_cont_sum'] - (deribit_df['forward_price']/deribit_df['strike_cutoff'] - 1)**2
    ivar_den = deribit_df['tte']
    deribit_df['dvar'] = (ivar_num / ivar_den).replace([np.inf, -np.inf], np.nan)
    deribit_df['dvol'] = np.sqrt(deribit_df['dvar'])
    return deribit_df


def compute_2perp_vols(deribit_df):
    deribit_df = filter_to_otm_options(deribit_df)
    
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    deribit_df['underlying_price'] = grouped_df['underlying_price'].transform('median')
    deribit_df = grouped_df.apply(lambda g: get_strike_width_and_prices(g, for_2perp=True)).\
        reset_index(drop=True)
    strike_cont = deribit_df['avg_instrument_price']*deribit_df['strike_width']*deribit_df['forward_price']
    deribit_df['2perp_strike_cont'] = 2*strike_cont
    
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    deribit_df['2perp_strike_cont_sum'] = grouped_df['2perp_strike_cont'].transform('sum') 
    
    deribit_df['2perp_rep_cost'] = deribit_df['strike_cutoff']**2 + deribit_df['2perp_strike_cont_sum'] + \
        2*deribit_df['strike_cutoff']*(deribit_df['forward_price'] - deribit_df['strike_cutoff'])
        
    ivar_num = np.log(deribit_df['2perp_rep_cost'] / deribit_df['underlying_price']**2)
    ivar_den = deribit_df['tte']
    deribit_df['2perp_var'] = (ivar_num / ivar_den).replace([np.inf, -np.inf], np.nan)
    deribit_df['2perp_vol'] = np.sqrt(deribit_df['2perp_var'])    
    return deribit_df


def compute_smile_avg_vols(deribit_df, moneyness_max=0):
    deribit_df = filter_to_otm_options(deribit_df)

    def _get_smile_avg_vol(date_df):
        moneyness_num = np.log(date_df['strike_price']/date_df['forward_price'])
        moneyness_den = date_df['mark_iv']*np.sqrt(date_df['tte'])
        moneyness_series = (moneyness_num / moneyness_den).replace([np.inf, -np.inf], np.nan)
        iv_series = date_df['mark_iv'].loc[~moneyness_series.isnull()]
        moneyness_series = moneyness_series.loc[~moneyness_series.isnull()]
        if len(moneyness_series) == 0:
            date_df['smile_avg_var'] = np.nan
            date_df['smile_avg_vol'] = np.nan
            return date_df
        
        interp_func = interp1d(moneyness_series, iv_series, fill_value='extrapolate')
        m_ivs = np.asarray(
            [interp_func(m) for m in np.arange(-moneyness_max, moneyness_max + 0.01, step=0.01)]
        )
        date_df['smile_avg_var'] = (m_ivs**2).mean()
        date_df['smile_avg_vol'] = np.sqrt(date_df['smile_avg_var'])
        return date_df
    
    grouped_df = deribit_df.groupby(['freq_timestamp', 'tte'])
    deribit_df = grouped_df.apply(_get_smile_avg_vol).reset_index(drop=True)
    return deribit_df


def compute_time_target_vol(deribit_df, target_tte_days = 17.5, vol_type = 'dvol'):
    target_tte = target_tte_days / 365.
    
    higher_tte = deribit_df.groupby('freq_timestamp')['tte'].\
        apply(lambda s: s[s > target_tte].min()).rename('higher_tte')
    lower_tte = deribit_df.groupby('freq_timestamp')['tte'].\
        apply(lambda s: s[s <= target_tte].max()).rename('lower_tte')
    deribit_df = deribit_df.join(higher_tte, on=['freq_timestamp'])
    deribit_df = deribit_df.join(lower_tte, on=['freq_timestamp'])
    
    higher_null = deribit_df['higher_tte'].isnull()
    lower_null = deribit_df['lower_tte'].isnull()
    
    if vol_type == 'dvol':
        var_name = 'dvar'
    elif vol_type == '2perp_vol':
        var_name = '2perp_var'
    elif vol_type == 'smile_avg_vol':
        var_name = 'smile_avg_var'
    
    higher_var_name = 'higher_{}'.format(var_name)
    lower_var_name = 'lower_{}'.format(var_name)
    
    var_df = pd.DataFrame(index=deribit_df[['freq_timestamp', 'tte']], data=deribit_df[var_name].values)
    var_dict = var_df.groupby(var_df.index).first().squeeze().to_dict()
    deribit_df[higher_var_name] = deribit_df.set_index(['freq_timestamp', 'higher_tte']).index.map(var_dict)
    deribit_df[lower_var_name] = deribit_df.set_index(['freq_timestamp', 'lower_tte']).index.map(var_dict)    
    
    deribit_df['higher_weight'] = (target_tte - deribit_df['lower_tte'])*deribit_df['higher_tte']
    deribit_df['lower_weight'] = (deribit_df['higher_tte'] - target_tte)*deribit_df['lower_tte']
    deribit_df['tte_diff'] = deribit_df['higher_tte'] - deribit_df['lower_tte']
    deribit_df['higher_weight'] /= deribit_df['tte_diff']
    deribit_df['lower_weight'] /= deribit_df['tte_diff']
    
    name = '{}_{}_days'.format(vol_type, target_tte_days)
    higher_w_var = deribit_df['higher_weight']*deribit_df[higher_var_name]
    lower_w_var = deribit_df['lower_weight']*deribit_df[lower_var_name]
    deribit_df[name] = np.sqrt(higher_w_var + lower_w_var)*np.sqrt(1 / target_tte)
    
    deribit_df.loc[higher_null, name] = np.sqrt(deribit_df.loc[higher_null, lower_var_name])
    deribit_df.loc[lower_null, name] = np.sqrt(deribit_df.loc[lower_null, higher_var_name])
    
    drop_cols = [
        'higher_tte', 'lower_tte', 'higher_weight', 'lower_weight', 'tte_diff', higher_var_name, lower_var_name
    ]
    deribit_df = deribit_df.drop(drop_cols, axis=1)
    return deribit_df


def compute_osqth_iv(squeeth_df, funding_period_days=17.5, osqth_norm=10_000):
    squeeth_price = squeeth_df['usdc_osqth']*(osqth_norm / squeeth_df['new_norm_factor'])
    eth_price = squeeth_df['usdc_eth']
    implied_daily_funding = np.log(squeeth_price/(eth_price**2)) / funding_period_days
    squeeth_df['osqth_iv'] = np.sqrt(implied_daily_funding*365)
    return squeeth_df


if __name__ == "__main__":
    deribit_df = pd.read_csv(STRATEGY_DATA_PATH + '/deribit_strategy_df.csv', index_col=0)
    squeeth_df = pd.read_csv(STRATEGY_DATA_PATH + '/squeeth_strategy_df.csv', index_col=0)

    deribit_df = compute_instrument_prices(deribit_df)
    deribit_df = compute_forward_prices(deribit_df)
    deribit_df = compute_strike_cutoffs(deribit_df)
    deribit_df = compute_dvols(deribit_df)
    deribit_df = compute_2perp_vols(deribit_df)
    deribit_df = compute_smile_avg_vols(deribit_df, moneyness_max=0)
    deribit_df = compute_time_target_vol(deribit_df, 17.5, 'dvol')
    deribit_df = compute_time_target_vol(deribit_df, 17.5, '2perp_vol')
    deribit_df = compute_time_target_vol(deribit_df, 17.5, 'smile_avg_vol')

    squeeth_df = compute_osqth_iv(squeeth_df)

    deribit_df.to_csv(STRATEGY_DATA_PATH + '/deribit_backtest_df.csv')
    squeeth_df.to_csv(STRATEGY_DATA_PATH + '/squeeth_backtest_df.csv')
