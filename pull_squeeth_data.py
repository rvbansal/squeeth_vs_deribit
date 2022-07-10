import json
import requests
from web3 import Web3


RAW_DOWNLOAD_PATH = './squeeth_datasets'
START_BLOCK = 13982541 # Start of Squeeth Controller contract
END_BLOCK = 14927827 # Last date of analysis

ALCHEMY_KEY = ''
BLOCK_CHUNK = 2000

NORM_FACTOR_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0x339e53729b0447795ff69e70a74fed98fc7fef6fe94b7521099b32f0f8de4822"
    ] 
}

MINT_SHORT_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0xb19fa182730a088464dad0e9e0badeb470d0d8d937d854f5caf15c6ad1992c36"
    ]
}

BURN_SHORT_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0xea19ffc45b48de6d95594aacff7214dd24595fdb0c60e98c8f0b269058c2f792"
    ]
}

DEPOSIT_COLLATERAL_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0x3ca13b7aab12bad7472691fe558faa6b25e99099824a0070a88bd5aa84be610f"
    ]
}

WITHDRAW_COLLATERAL_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0x627a692d5a03ab34732c0d2aa319f3ecdebdc4528f383eabcb25441dc0a70cfb"
    ]
}

DEPOSIT_UNI_POSITION_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0x3917c2f26ce18614e3aedd1289da672ef6563c5c295f49e9b1697ae0ad315562"
    ]
}

WITHDRAW_UNI_POSITION_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0xe59f38fa1264fc25c9f0185eee136eaf810d90b8e7293b342e4037c68720177a"
    ]
}

LIQUIDATE_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0x158ba9ab7bbbd08eeffa4753bad41f4d450e24831d293427308badf3eadd8c76"
    ]
}

OPEN_VAULT_DATA = {
    "address": "0x64187ae08781B09368e6253F9E94951243A493D5",
    "topics": [
        "0x25ff1e0131762176a9084e92880f880f39d6d0e62134607f37e631efe1bad871"
    ]
}

OSQTH_ETH_SWAPS_DATA = {
    "address": "0x82c427adfdf2d245ec51d8046b41c4ee87f0d29c",
    "topics": [
        "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
    ]
}

OSQTH_ETH_MINT_DATA = {
    "address": "0x82c427adfdf2d245ec51d8046b41c4ee87f0d29c",
    "topics": [
        "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde"
    ]
}

OSQTH_ETH_BURN_DATA = {
    "address": "0x82c427adfdf2d245ec51d8046b41c4ee87f0d29c",
    "topics": [
        "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c"
    ]
}

OSQTH_ETH_COLLECT_DATA = {
    "address": "0x82c427adfdf2d245ec51d8046b41c4ee87f0d29c",
    "topics": [
        "0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0"
    ]
}

ETH_USDC_SWAPS_DATA = {
    "address": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
    "topics": [
        "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
    ]
}


def pull_and_save_contract_data(
  request_data, file_name, start_block=START_BLOCK, end_block=END_BLOCK, block_chunk=BLOCK_CHUNK
):
    w3 = Web3(Web3.HTTPProvider(ALCHEMY_KEY))
    request_data = {
      "jsonrpc": "2.0", "id": 0, "method": "eth_getLogs", "params": [request_data]
    }
    output_data = []
    for start_block in range(start_block, end_block, block_chunk + 1):
        end_block = start_block + block_chunk
        request_data["params"][0]["fromBlock"] = w3.toHex(start_block)
        request_data["params"][0]["toBlock"] = w3.toHex(end_block)
        response = requests.post(
          ALCHEMY_KEY, headers = {"Content-Type": "application/json"}, json=request_data
        )
        data = response.json()
        data = data['result']
        output_data.extend(data)

    with open(RAW_DOWNLOAD_PATH + '/{}.json'.format(file_name), 'w') as outfile:
        json.dump(output_data, outfile)


def pull_and_save_block_timestamps(start_block=START_BLOCK, end_block=END_BLOCK):
    w3 = Web3(Web3.HTTPProvider(ALCHEMY_KEY))
    output_dict = {
      block: w3.eth.get_block(block)['timestamp'] for block in range(start_block, end_block + 1)
    }
    with open(RAW_DOWNLOAD_PATH + '/block_timestamps.json', 'w') as outfile:
        json.dump(output_dict, outfile)


if __name__ == "__main__":
    pull_and_save_contract_data(NORM_FACTOR_DATA, 'norm_factor')
    pull_and_save_contract_data(MINT_SHORT_DATA, 'mint_short')
    pull_and_save_contract_data(BURN_SHORT_DATA, 'burn_short')
    pull_and_save_contract_data(DEPOSIT_COLLATERAL_DATA, 'deposit_collateral')
    pull_and_save_contract_data(WITHDRAW_COLLATERAL_DATA, 'withdraw_collateral')
    pull_and_save_contract_data(DEPOSIT_UNI_POSITION_DATA, 'deposit_uni_position')
    pull_and_save_contract_data(WITHDRAW_UNI_POSITION_DATA, 'withdraw_uni_position')
    pull_and_save_contract_data(LIQUIDATE_DATA, 'liquidate')
    pull_and_save_contract_data(OPEN_VAULT_DATA, 'open_vault')
    pull_and_save_contract_data(OSQTH_ETH_SWAPS_DATA, 'osqth_eth')
    pull_and_save_contract_data(OSQTH_ETH_MINT_DATA, 'osqth_eth_mint')
    pull_and_save_contract_data(OSQTH_ETH_BURN_DATA, 'osqth_eth_burn')
    pull_and_save_contract_data(OSQTH_ETH_COLLECT_DATA, 'osqth_eth_collect')
    pull_and_save_contract_data(ETH_USDC_SWAPS_DATA, 'eth_usdc')

    pull_and_save_block_timestamps(START_BLOCK, END_BLOCK)
