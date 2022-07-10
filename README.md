I have removed the raw Deribit and Squeeth datasets. It would take quite a while, but to run the data processing pipeline you need Tardis and Alchemy API keys and then you can run:

```
python pull_deribit_data.py
python pull_squeeth_data.py
python process_deribit_data.py
python process_squeeth_data.py
```

To run the core modeling code, you should do:

```
python build_squeeth_ts_df.py
python build_deribit_ts_df.py
python iv_calculator.py
```

Check ```backtest_runs.ipynb``` for examples on how to use ```backtester.py``` to run and analyze backtests with different settings.

The other notebooks are mostly for plotting and simple simulations to verify the Squeeth pricing model.