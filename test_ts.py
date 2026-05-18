import tensorstore as ts

orbax_dir = "gs://tinymath-reason-data-himanshu/checkpoints/tinymath-tiny-test/checkpoints/1/items"

kvs = ts.KvStore.open({
    'driver': 'ocdbt',
    'base': orbax_dir,
}).result()

keys = kvs.list().result()
for k in keys:
    k_str = k.decode()
    if k_str.endswith("zarr.json"):
        arr_path = k_str.replace("/zarr.json", "")
        if "opt_state" in arr_path or "step" in arr_path: continue
        
        print(f"Opening {arr_path} with zarr3...")
        try:
            dataset = ts.open({
                'driver': 'zarr3',
                'kvstore': {'driver': 'ocdbt', 'base': orbax_dir},
                'path': arr_path
            }).result()
            
            arr = dataset.read().result()
            print("Shape:", arr.shape, "Dtype:", arr.dtype)
            break
        except Exception as e:
            print("Error:", e)
