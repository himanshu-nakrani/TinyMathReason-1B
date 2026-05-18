import tensorstore as ts
import sys

base = "tinymath-reason-data-himanshu/checkpoints/tinymath-tiny-test/checkpoints/1/items"

kvs = ts.KvStore.open({
    'driver': 'ocdbt',
    'base': f'gs://{base}/',
}).result()

for k in kvs.list().result():
    decoded = k.decode()
    if 'zarr.json' in decoded:
        print(decoded)
