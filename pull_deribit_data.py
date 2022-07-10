from tardis_dev import datasets, get_exchange_details
import nest_asyncio

nest_asyncio.apply()

EXCHANGE = 'deribit'
DATA_TYPES = ['options_chain']
FROM_DATE = '2022-01-10'
TO_DATE = '2022-06-05'
SYMBOLS = ['OPTIONS']
API_KEY = ''
DOWNLOAD_PATH = './deribit_datasets'


if __name__ == "__main__":
    datasets.download(
        exchange = EXCHANGE,
        data_types = DATA_TYPES,
        from_date =  FROM_DATE,
        to_date = TO_DATE,
        symbols = SYMBOLS,
        api_key = API_KEY,
        download_dir = DOWNLOAD_PATH,
    )
