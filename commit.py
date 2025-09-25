import bittensor as bt

# Configure your wallet and the subtensor
wallet = bt.wallet(name="btwall_1", hotkey="bthotkey_1")
subtensor = bt.subtensor(network="finney")

my_hotkey = "5Hme13v5gL5CYA5fZuYTdwYpTVwdPenVZ3sDoQ8gX2qoWrmx" 

# The public URL where the validator can download your payload file.
# The final path component MUST match your hotkey.
public_url = f"http://213.173.105.105:16105/{my_hotkey}" 

# Commit the URL on-chain
res = subtensor.commit(wallet=wallet, netuid=123, data=public_url) # Use the correct netuid

print('commit result---', res)