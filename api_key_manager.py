import os
import itertools
from dotenv import load_dotenv

load_dotenv()
num_keys = int(os.getenv("NUM_NVIDIA_API_KEYS", 0))
api_keys = [os.getenv(f"NVIDIA_API_KEY_{i}") for i in range(1, num_keys + 1)] if num_keys else [os.getenv("NVIDIA_API_KEY")]
api_keys = [k for k in api_keys if k]
if not api_keys:
    raise ValueError("No NVIDIA API keys found in .env")

key_cycle = itertools.cycle(api_keys)
get_next_api_key = lambda: next(key_cycle)
