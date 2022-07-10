import pandas as pd
import numpy as np
from scipy.stats import norm


STRATEGY_DATA_PATH = './strategy_dfs'


class Portfolio:
    def __init__(self, assets, last_rebalance=pd.Timestamp.max.tz_localize('utc')):
        self.assets = assets
        self.last_rebalance = last_rebalance
    
    def num_units_sum(self, asset_type):
        asset_objs = self.assets[asset_type]
        return sum([a.num_units for a in asset_objs])
    
    def num_units_by_type(self):
        if len(self.assets) == 0:
            return 0
        asset_type_sums = {at: self.num_units_sum(at) for at in self.assets.keys()}        
        return asset_type_sums

    def asset_type_sum(self, name, asset_type, abs_val=False):
        asset_objs = self.assets[asset_type]
        if abs_val == True:
            return sum([getattr(a, name)*abs(a.num_units) for a in asset_objs])
        return sum([getattr(a, name)*a.num_units for a in asset_objs])
    
    def portfolio_sum_by_type(self, name, abs_val=False):
        if len(self.assets) == 0:
            return 0
        asset_type_sums = {at: self.asset_type_sum(name, at, abs_val) for at in self.assets.keys()}        
        return asset_type_sums
    
    def portfolio_sum(self, name, abs_val=False):
        if len(self.assets) == 0:
            return 0
        return sum(self.portfolio_sum_by_type(name, abs_val).values())
    
    @property
    def delta(self):
        return self.portfolio_sum(name='delta')
    
    @property
    def gamma(self):
        return self.portfolio_sum(name='gamma')
    
    @property
    def vega(self):
        return self.portfolio_sum(name='vega')
    
    @property
    def theta(self):
        return self.portfolio_sum(name='theta')
    
    @property
    def tcost(self):
        return self.portfolio_sum(name='tcost', abs_val=True)
    
    @property
    def unrealized_pnl(self):
        return self.portfolio_sum(name='unrealized_pnl')

    @property
    def cumulative_unrealized_pnl(self):
        return self.portfolio_sum(name='cumulative_unrealized_pnl')
    
    @property
    def capital(self):
        if len(self.assets) == 0:
            return 0
        eth_collateral_position = abs(self.asset_type_sum(name='curr_price', asset_type='eth_collateral'))
        option_position = abs(self.asset_type_sum(name='curr_price', asset_type='options'))
        eth_delta_hedge_position = abs(self.asset_type_sum(name='curr_price', asset_type='eth_delta_hedge'))
        return eth_collateral_position + option_position + eth_delta_hedge_position
    
    @property
    def collateral_ratio(self):
        if 'eth_collateral' not in self.assets:
            return np.nan
        if 'osqth' not in self.assets:
            return np.nan
        cr_numerator = self.num_units_sum('eth_collateral')
        osqth_debt =  -self.num_units_sum('osqth')
        osqth_asset = self.assets['osqth'][0]
        cr_denominator = osqth_debt*(osqth_asset.norm_factor/osqth_asset.osqth_norm)*osqth_asset.eth_price        
        cr = cr_numerator / cr_denominator
        return cr

    def step(self, date):
        for asset_type in self.assets:
            for asset in self.assets[asset_type]:
                asset.step(date)


class AssetPosition:
    def __init__(self, t, ts_df, num_units, tcost_multiplier, price_col):
        self.ts_df = ts_df
        self.num_units = num_units
        self.price_col = price_col
        self.tcost_multiplier = tcost_multiplier
        self.open_position_eod(t)
    
    def open_position_eod(self, date):
        self.start_date = date
        self.curr_date = date
        self.end_date = pd.Timestamp.max.tz_localize('utc')
        self.curr_price = self.ts_df.loc[self.curr_date, self.price_col]
        sign = -1 if self.num_units > 0 else 1
        self.unrealized_pnl = sign*self.tcost
        self.cumulative_unrealized_pnl = sign*self.tcost
    
    def close_position_eod(self, date):
        self.end_date = date
        self.step(date)
        sign = -1 if self.num_units > 0 else 1
        self.unrealized_pnl += sign*self.tcost
        self.cumulative_unrealized_pnl += sign*self.tcost
    
    @property
    def delta(self):
        pass
    
    @property
    def gamma(self):
        pass
    
    @property
    def vega(self):
        pass
    
    @property
    def theta(self):
        pass
    
    @property
    def tcost(self):
        if self.curr_date == self.start_date:
            return self.tcost_multiplier*self.compute_tcost()
        elif self.curr_date == self.end_date:
            return self.tcost_multiplier*self.compute_tcost()
        return 0
    
    def compute_tcost(self):
        pass
    
    def step(self, date):
        new_price = self.ts_df.loc[date, self.price_col]
        self.unrealized_pnl = new_price - self.curr_price
        self.cumulative_unrealized_pnl += new_price - self.curr_price
        self.curr_date = date
        self.curr_price = new_price


class DeribitCall(AssetPosition):
    def __init__(self, t, ts_df, num_units=1, tcost_multiplier=1, price_col='opt_price_usdc'):
        self.r = 0
        self.q = 0
        self.take_fee_perc = 0.0003
        self.max_fee_perc = 0.125
        self.market_impact_perc = 0.05
        super().__init__(t, ts_df, num_units, tcost_multiplier, price_col)
    
    def open_position_eod(self, date):
        self.time_to_expiry = self.ts_df.loc[date, 'tte']
        self.strike_price = self.ts_df.loc[date, 'strike_price']
        self.eth_price = self.ts_df.loc[date, 'underlying_price']
        self.iv = self.ts_df.loc[date, 'mark_iv']
        self.expiration_date = self.ts_df.loc[date, 'expiration']
        super().open_position_eod(date)

    @property
    def d1(self):
        numerator = np.log(self.eth_price/self.strike_price) + \
            self.time_to_expiry*(self.r - self.q + self.iv**2/2)
        denominator = self.iv*np.sqrt(self.time_to_expiry)
        return numerator / denominator  
    
    @property
    def d2(self):
        return self.d1 - self.iv*np.sqrt(self.time_to_expiry)
    
    def _price(self):
        term1 = self.eth_price*np.exp(-self.q*self.time_to_expiry)*norm.cdf(self.d1)
        term2 = self.strike_price*np.exp(-self.r*self.time_to_expiry)*norm.cdf(self.d2)
        return term1 - term2
    
    @property
    def delta(self):
        return np.exp(-self.q*self.time_to_expiry)*norm.cdf(self.d1)
    
    @property
    def gamma(self):
        numerator = np.exp(-self.q*self.time_to_expiry)*norm.pdf(self.d1)
        denominator = self.eth_price*self.iv*np.sqrt(self.time_to_expiry)
        return numerator / denominator
    
    @property
    def vega(self):
        return self.eth_price*np.exp(-self.q*self.time_to_expiry)*np.sqrt(self.time_to_expiry)*\
            norm.pdf(self.d1) / 100.
    
    @property
    def theta(self):
        term1_numerator = self.eth_price*self.iv*np.exp(-self.q*self.time_to_expiry)*\
            norm.pdf(self.d1)
        term1_denominator = 2*np.sqrt(self.time_to_expiry)
        term1 = term1_numerator / term1_denominator
        term2 = self.r*self.strike_price*np.exp(-self.r*self.time_to_expiry)*norm.cdf(self.d2)
        term3 = self.q*self.eth_price*np.exp(-self.q*self.time_to_expiry)*norm.cdf(self.d1)
        return (-term1 - term2 + term3) / 365
    
    @property
    def moneyness(self):
        numerator = np.log(self.eth_price/self.strike_price) + self.r*self.time_to_expiry
        denominator = self.iv*np.sqrt(self.time_to_expiry)
        moneyness = numerator / denominator
        return moneyness
            
    def compute_tcost(self):
        taker_fee = self.take_fee_perc*self.eth_price
        max_possible_fee = self.max_fee_perc*self.curr_price
        price_impact = self.curr_price*self.market_impact_perc
        return abs(min(taker_fee, max_possible_fee)) + abs(price_impact)
    
    def step(self, date):
        if date >= self.expiration_date:
            print('Option expired on {}. Today is {}'.format(self.expiration_date, date))
            return
        super().step(date)
        self.time_to_expiry = self.ts_df.loc[date, 'tte']
        self.eth_price = self.ts_df.loc[date, 'underlying_price']
        self.iv = self.ts_df.loc[date, 'mark_iv']


class DeribitPut(DeribitCall):
    def __init__(self, t, ts_df, num_units=1, tcost_multiplier=1, price_col='opt_price_usdc'):
        super().__init__(t, ts_df, num_units, tcost_multiplier, price_col)
    
    def _price(self):
        term1 = self.eth_price*np.exp(-self.q*self.time_to_expiry)*norm.cdf(-self.d1)
        term2 = self.strike_price*np.exp(-self.r*self.time_to_expiry)*norm.cdf(-self.d2)
        return term2 - term1
    
    @property
    def delta(self):
        return np.exp(-self.q*self.time_to_expiry)*(norm.cdf(self.d1) - 1)
    
    @property
    def theta(self):
        term1_numerator = self.eth_price*self.iv*np.exp(-self.q*self.time_to_expiry)*\
            norm.pdf(self.d1)
        term1_denominator = 2*np.sqrt(self.time_to_expiry)
        term1 = term1_numerator / term1_denominator
        term2 = self.r*self.strike_price*np.exp(-self.r*self.time_to_expiry)*norm.cdf(-self.d2)
        term3 = self.q*self.eth_price*np.exp(-self.q*self.time_to_expiry)*norm.cdf(-self.d1)
        return (-term1 + term2 - term3) / 365


class Osqth(AssetPosition):
    def __init__(self, t, ts_df, num_units=1, tcost_multiplier=1, price_col='usdc_osqth'):
        self.osqth_norm = 10_000
        self.funding_days = 17.5
        self.funding_period = self.funding_days / 365.
        self.osqth_swap_fee_perc = 0.018
        super().__init__(t, ts_df, num_units, tcost_multiplier, price_col)

    def open_position_eod(self, date):
        self.eth_price = self.ts_df.loc[date, 'usdc_eth']
        self.norm_factor = self.ts_df.loc[date, 'new_norm_factor']
        super().open_position_eod(date)
    
    @property
    def squeeth_price(self):
        return (self.osqth_norm/self.norm_factor)*self.curr_price

    @property
    def implied_daily_funding(self):
        return np.log(self.squeeth_price/(self.eth_price**2)) / self.funding_days
    
    @property
    def iv(self):
        return np.sqrt(self.implied_daily_funding*365.)
    
    def _implied_squeeth_price(self):
        return self.eth_price**2*np.exp(self.iv**2*self.funding_period)

    def _price(self):
        return (self.norm_factor/self.osqth_norm)*self._implied_squeeth_price()
    
    @property
    def delta(self):
        squeeth_delta = 2*self.eth_price*np.exp(self.iv**2*self.funding_period)
        return (self.norm_factor/self.osqth_norm)*squeeth_delta
    
    @property
    def gamma(self):
        squeeth_gamma = 2*np.exp(self.iv**2*self.funding_period)
        return (self.norm_factor/self.osqth_norm)*squeeth_gamma
    
    @property
    def vega(self):
        squeeth_vega = 2*self.iv*self.funding_period*self._implied_squeeth_price() / 100
        return (self.norm_factor/self.osqth_norm)*squeeth_vega
    
    @property
    def theta(self):
        squeeth_theta = -(self.iv**2)*self._implied_squeeth_price() / 365
        return (self.norm_factor/self.osqth_norm)*squeeth_theta

    def compute_tcost(self):
        return abs(self.curr_price*self.osqth_swap_fee_perc)

    def step(self, date):
        super().step(date)
        self.eth_price = self.ts_df.loc[date, 'usdc_eth']
        self.norm_factor = self.ts_df.loc[date, 'new_norm_factor']


class Eth(AssetPosition):
    def __init__(self, t, ts_df, num_units=1, tcost_multiplier=1, price_col='usdc_eth'):
        self.trade_fee_perc = 0.001
        super().__init__(t, ts_df, num_units, tcost_multiplier, price_col)
    
    @property
    def delta(self):
        return 1
    
    @property
    def gamma(self):
        return 0
    
    @property
    def vega(self):
        return 0
    
    @property
    def theta(self):
        return 0

    def compute_tcost(self):
        return abs(self.curr_price*self.trade_fee_perc)


class StrategyRunner:
    def __init__(
        self,
        squeeth_strategy_df,
        deribit_strategy_df,
        deribit_iv_col = '2perp_vol_17.5_days',
        target_tte_days = 17.5,
        ivol_ratio_trigger = 1.3,
        ivol_close_ratio_trigger = 1.3,
        ivol_smooth_halflife = 3,
        target_osqth_exposure = 50_000,
        target_collateral_ratio = 2,
        hedge_rebalance_freq_days = 10,
        delta_rebalance_trigger = 5,
        gamma_rebalance_trigger = 0.01,
        collateral_ratio_triggers = [1.75, 2.25],
        expiration_buffer_days = 2,
        tcost_multiplier = 1
    ):
        self.squeeth_strategy_df = squeeth_strategy_df.copy()
        self.deribit_strategy_df = deribit_strategy_df.copy().set_index('freq_timestamp')
        
        self.deribit_iv_col = deribit_iv_col
        self.ivol_ratio_trigger = ivol_ratio_trigger
        self.ivol_close_ratio_trigger = ivol_ratio_trigger
        self.ivol_smooth_halflife = ivol_smooth_halflife
        self.target_tte_days = target_tte_days
        assert self.ivol_close_ratio_trigger <= self.ivol_ratio_trigger, \
            'IV close ratio trigger > IV ratio trigger!'
        
        self.target_osqth_exposure = target_osqth_exposure
        self.target_collateral_ratio = target_collateral_ratio
        
        self.hedge_rebalance_freq_days = hedge_rebalance_freq_days
        self.expiration_buffer_days = expiration_buffer_days
        self.delta_rebalance_trigger = delta_rebalance_trigger
        self.gamma_rebalance_trigger = gamma_rebalance_trigger
        self.collateral_ratio_triggers = collateral_ratio_triggers
        self.tcost_multiplier = tcost_multiplier

        self.set_up()
        
    def set_up(self):
        self.squeeth_strategy_df.index = pd.to_datetime(self.squeeth_strategy_df.index, unit='ns')
        self.deribit_strategy_df.index = pd.to_datetime(self.deribit_strategy_df.index, unit='ns')

        for col in ['expiration', 'timestamp', 'local_timestamp']:
            self.deribit_strategy_df[col] = pd.to_datetime(self.deribit_strategy_df[col], unit='ns')

        self.deribit_iv = self.deribit_strategy_df[self.deribit_iv_col].drop_duplicates().\
            fillna(method='ffill').ewm(self.ivol_smooth_halflife, min_periods=1).mean()
        self.osqth_iv = self.squeeth_strategy_df['osqth_iv'].\
            fillna(method='ffill').ewm(self.ivol_smooth_halflife, min_periods=1).mean()
        self.iv_ratio = self.osqth_iv / self.deribit_iv

        first_date = self.iv_ratio.first_valid_index()
        last_date = self.iv_ratio.last_valid_index()
        self.dates = pd.date_range(first_date, last_date, freq='H')
        self.iv_ratio = self.iv_ratio.reindex(self.dates)
        self.squeeth_strategy_df = self.squeeth_strategy_df.reindex(self.dates)
        
        self.portfolio = Portfolio(assets={})
        self.active_portfolio = False
    
    def open_position(self, t):
        # Find amount of osqth to get ~$50K short exposure
        osqth_asset = Osqth(t, self.squeeth_strategy_df, 1, self.tcost_multiplier)
        osqth_asset.num_units = -self.target_osqth_exposure / osqth_asset.curr_price
        
        # Find options closest to 17.5 days to expiry
        deribit_options = self.deribit_strategy_df.loc[t]
        avail_tte = deribit_options['tte'].unique()
        selected_tte = avail_tte[(avail_tte > self.target_tte_days / 365.)].min()
        tte_options = deribit_options[deribit_options['tte'] == selected_tte]
        tte_options = tte_options.set_index(['symbol', 'type'])
        
        # Weight options per strike width to match osqth replicating portfolio
        weights = tte_options['strike_width'] / tte_options['strike_width'].sum()
        options_assets = []
        for info, lots in weights.items():
            symbol, opt_type = info
            opt_data = self.deribit_strategy_df[self.deribit_strategy_df['symbol'] == symbol]
            opt_data = opt_data.reindex(self.dates).fillna(method='ffill')
            if opt_type == 'call':
                option = DeribitCall(t, opt_data, lots, self.tcost_multiplier)
            elif opt_type == 'put':
                option = DeribitPut(t, opt_data, lots, self.tcost_multiplier)
            options_assets.append(option)
        
        # Weight options to gamma hedge the short osqth exposure
        total_osqth_gamma = abs(osqth_asset.gamma*osqth_asset.num_units)
        total_option_gamma = sum([o.gamma*o.num_units for o in options_assets])
        gamma_scalar = total_osqth_gamma / total_option_gamma
        for option in options_assets:
            option.num_units *= gamma_scalar

        # Compute required eth collateral
        osqth_debt = -osqth_asset.num_units
        cr_denominator = osqth_debt*(osqth_asset.norm_factor/osqth_asset.osqth_norm)*osqth_asset.eth_price
        eth_collateral = self.target_collateral_ratio*cr_denominator
        eth_collateral_asset = Eth(t, self.squeeth_strategy_df, eth_collateral, self.tcost_multiplier)
        
        # Build initial portfolio
        self.portfolio = Portfolio(assets={}, last_rebalance=t)
        self.portfolio.assets['osqth'] = [osqth_asset]
        self.portfolio.assets['eth_collateral'] = [eth_collateral_asset]
        self.portfolio.assets['options'] = options_assets
        
        # Get total portfolio delta and add residual delta hedge
        eth_hedge_asset = Eth(t, self.squeeth_strategy_df, -self.portfolio.delta, self.tcost_multiplier)
        self.portfolio.assets['eth_delta_hedge'] = [eth_hedge_asset]
        
        self.active_portfolio = True
        
    def check_rebalance_need(self, t):
        min_cr, max_cr = self.collateral_ratio_triggers
        if not (min_cr <= self.portfolio.collateral_ratio <= max_cr):
            print('Rebalance on {} due to {}.'.format(t, 'collaterization ratio'))
            return (True, 'cr')
        if abs(self.portfolio.delta) >= self.delta_rebalance_trigger:
            print('Rebalance on {} due to {}.'.format(t, 'delta'))
            return (True, 'delta')
        if abs(self.portfolio.gamma) >= self.gamma_rebalance_trigger:
            print('Rebalance on {} due to {}.'.format(t, 'gamma'))
            return (True, 'gamma')
        if (t - self.portfolio.last_rebalance) >= pd.Timedelta(self.hedge_rebalance_freq_days, "d"):
            print('Rebalance on {} due to {}.'.format(t, 'time'))
            return (True, 'time')        
        return (False, 'ignore')
    
    def rebalance_collateral_ratio(self, t):
        cr = self.portfolio.collateral_ratio
        curr_eth_units = self.portfolio.num_units_sum('eth_collateral')
        diff_ratio = cr / self.target_collateral_ratio
        diff_amt = (1. / diff_ratio)*curr_eth_units - curr_eth_units
        eth_collateral_asset = Eth(t, self.squeeth_strategy_df, diff_amt, self.tcost_multiplier)
        self.portfolio.assets['eth_collateral'].append(eth_collateral_asset)
    
    def rebalance_delta(self, t):
        portfolio_delta = self.portfolio.delta
        eth_hedge_quantity = abs(portfolio_delta)
        sign = -1 if portfolio_delta > 0 else 1
        eth_hedge_asset = Eth(t, self.squeeth_strategy_df, sign*eth_hedge_quantity, self.tcost_multiplier)
        self.portfolio.assets['eth_delta_hedge'].append(eth_hedge_asset)
    
    def rebalance_gamma(self, t):
        portfolio_gamma = self.portfolio.gamma
        osqth_gamma_t = self.portfolio.assets['osqth'][0].gamma
        osqth_quantity = abs(portfolio_gamma / osqth_gamma_t)
        sign = -1 if portfolio_gamma > 0 else 1
        osqth_asset = Osqth(t, self.squeeth_strategy_df, sign*osqth_quantity, self.tcost_multiplier)
        self.portfolio.assets['osqth'].append(osqth_asset)
    
    def rebalance(self, t):
        rebalance_bool, trigger = self.check_rebalance_need(t)
        if rebalance_bool:
            self.portfolio.last_rebalance = t
            if trigger == 'cr' or trigger == 'delta':
                self.rebalance_collateral_ratio(t)
                self.rebalance_delta(t)
            elif trigger == 'gamma' or trigger == 'time':
                self.rebalance_gamma(t)
                self.rebalance_collateral_ratio(t)
                self.rebalance_delta(t)
    
    def check_options_expiration_trigger(self, t):
        expiration_date = self.portfolio.assets['options'][0].expiration_date
        if (expiration_date - t) <= pd.Timedelta(self.expiration_buffer_days, "d"):
            return True
        return False
    
    def close_position(self, t):
        for asset_type in self.portfolio.assets:
            for asset in self.portfolio.assets[asset_type]:
                asset.close_position_eod(t)
                
        self.close_position_unrealized_pnl = self.portfolio.unrealized_pnl
        self.close_position_realized_pnl = self.portfolio.cumulative_unrealized_pnl

        self.portfolio = Portfolio(assets={})
        self.active_portfolio = False
    
    def run(self):
        strategy_run_output = {}
        self.portfolio_cumulative_realized_pnl = 0
        
        for t in self.dates:
            self.close_position_realized_pnl = 0
            self.close_position_unrealized_pnl = 0
            
            if self.active_portfolio:
                if self.iv_ratio[t] >= self.ivol_close_ratio_trigger:
                    if self.check_options_expiration_trigger(t):
                        self.close_position(t)
                        self.open_position(t)
                    elif self.portfolio.capital < self.target_osqth_exposure*self.collateral_ratio_triggers[0]:
                        self.close_position(t)
                        self.open_position(t)
                    else:
                        self.portfolio.step(t)
                        self.rebalance(t)
                else:
                    print('CLOSE position on {}.'.format(t))
                    self.close_position(t)
            else:
                if self.iv_ratio[t] >= self.ivol_ratio_trigger:
                    print('OPEN position on {}.'.format(t))
                    self.open_position(t)
            
            self.portfolio_cumulative_realized_pnl += self.close_position_realized_pnl
                        
            strategy_run_output[t] = {
                'delta': self.portfolio.delta,
                'gamma': self.portfolio.gamma,
                'vega': self.portfolio.vega,
                'theta': self.portfolio.theta,
                'unrealized_pnl': self.portfolio.unrealized_pnl + self.close_position_unrealized_pnl,
                'realized_pnl': self.close_position_realized_pnl,
                'cumulative_realized_pnl': self.portfolio_cumulative_realized_pnl,
                'capital': self.portfolio.capital,
                'tcost': self.portfolio.tcost,
                'collateral_ratio': self.portfolio.collateral_ratio,
                'num_units_by_asset': self.portfolio.num_units_by_type(), 
                'delta_by_asset': self.portfolio.portfolio_sum_by_type('delta'),
                'gamma_by_asset': self.portfolio.portfolio_sum_by_type('gamma'),
                'vega_by_asset': self.portfolio.portfolio_sum_by_type('vega'),
                'theta_by_asset': self.portfolio.portfolio_sum_by_type('theta'),
                'tcost_by_asset': self.portfolio.portfolio_sum_by_type('tcost', abs_val=True),
                'position_size_by_asset': self.portfolio.portfolio_sum_by_type('curr_price'),
                'unrealized_pnl_by_asset': self.portfolio.portfolio_sum_by_type('unrealized_pnl'),
            }
        return strategy_run_output
