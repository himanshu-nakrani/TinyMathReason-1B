import tensorstore as ts
import asyncio
import numpy as np

async def main():
    base = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-tiny-test/checkpoints/1/items"
    kvs = await ts.KvStore.open({
        'driver': 'ocdbt',
        'base': base,
    })
    keys = await kvs.list()
    
    # Filter only parameters
    param_keys = [k.decode().replace('/zarr.json', '') for k in keys if 'params' in k.decode() and 'zarr.json' in k.decode()]
    
    res = {}
    for key in param_keys:
        try:
            dataset = await ts.open({
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'ocdbt',
                    'base': f"{base}/{key}"
                }
            })
            arr = await dataset.read()
            res[key] = arr
            print(f"Loaded {key} with shape {arr.shape}")
        except Exception as e:
            print(f"Failed to load {key}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
