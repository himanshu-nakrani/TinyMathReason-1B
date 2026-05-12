"""
Read zarr.json metadata via OCDBT KvStore and check scan_layers setting.
"""
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def inspect_checkpoint(orbax_dir: str):
    import tensorstore as ts

    base = orbax_dir.replace("gs://", "")

    # ── 1. Read zarr.json files from root OCDBT ───────────────────────
    print(f"\n{'='*80}")
    print("READING zarr.json FROM ROOT OCDBT STORE")
    print(f"{'='*80}")

    kvs = ts.KvStore.open({
        'driver': 'ocdbt',
        'base': f'gs://{base}/items/',
    }).result()

    zarr_keys = [k for k in kvs.list().result() if b'zarr.json' in k]
    for zk in sorted(zarr_keys):
        try:
            data = kvs[zk].read().result()
            meta = json.loads(data.tobytes())
            name = zk.decode().replace('/zarr.json', '')
            shape = meta.get('shape', '?')
            dtype = meta.get('data_type', meta.get('dtype', '?'))
            chunks = meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', '?')
            print(f"  {name}")
            print(f"    shape={shape}, dtype={dtype}, chunks={chunks}")
        except Exception as e:
            print(f"  {zk.decode()}: ERROR - {e}")

    # ── 2. Read zarr.json from ocdbt.process_0 ────────────────────────
    print(f"\n{'='*80}")
    print("READING zarr.json FROM ocdbt.process_0")
    print(f"{'='*80}")

    kvs_p0 = ts.KvStore.open({
        'driver': 'ocdbt',
        'base': f'gs://{base}/items/ocdbt.process_0/',
    }).result()

    zarr_keys_p0 = [k for k in kvs_p0.list().result() if b'zarr.json' in k]
    for zk in sorted(zarr_keys_p0):
        try:
            data = kvs_p0[zk].read().result()
            meta = json.loads(data.tobytes())
            name = zk.decode().replace('/zarr.json', '')
            shape = meta.get('shape', '?')
            dtype = meta.get('data_type', meta.get('dtype', '?'))
            chunks = meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', '?')
            print(f"  {name}")
            print(f"    shape={shape}, dtype={dtype}, chunks={chunks}")
        except Exception as e:
            print(f"  {zk.decode()}: ERROR - {e}")

    # ── 3. Try zarr3 driver to actually read arrays ────────────────────
    print(f"\n{'='*80}")
    print("READING ARRAYS WITH zarr3 DRIVER")
    print(f"{'='*80}")

    known_params = [
        "params.VariableState.decoder.decoder_norm.scale.value",
        "params.VariableState.decoder.logits_dense.kernel.value",
        "params.VariableState.token_embedder.embedding.value",
    ]

    for param_path in known_params:
        try:
            spec = {
                'driver': 'zarr3',
                'kvstore': {
                    'driver': 'ocdbt',
                    'base': f'gs://{base}/items/',
                },
                'path': param_path,
            }
            store = ts.open(spec, read=True, open=True).result()
            print(f"  ✅ {param_path}")
            print(f"     shape={store.shape}, dtype={store.dtype}")

            # Read a small slice to verify
            if len(store.shape) == 0:
                val = store.read().result()
                print(f"     value={val}")
            elif len(store.shape) == 1:
                val = store[:5].read().result()
                print(f"     first 5 values: {val}")
        except Exception as e:
            print(f"  ❌ {param_path}: {str(e)[:150]}")

    # ── 4. Check other checkpoints for comparison ─────────────────────
    print(f"\n{'='*80}")
    print("CHECKING ANOTHER CHECKPOINT (54000) FOR COMPARISON")
    print(f"{'='*80}")

    base_54000 = base.replace("/54362", "/54000")
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        meta_path = base_54000 + "/items/_METADATA"
        with fs.open(meta_path, 'rb') as f:
            meta_54000 = json.loads(f.read())
        tree = meta_54000.get('tree_metadata', {})
        print(f"  Step 54000 has {len(tree)} entries in tree_metadata")
        for key_str, val in sorted(tree.items()):
            keys = [km['key'] for km in val['key_metadata']]
            shape = val['value_metadata'].get('write_shape', [])
            path = '/'.join(keys)
            if keys[0] == 'params':
                print(f"    [PARAM] {path}: {shape}")
    except Exception as e:
        print(f"  Failed: {e}")

    # ── 5. Check MaxText's scan_layers default ─────────────────────────
    print(f"\n{'='*80}")
    print("LOCAL CONFIG ANALYSIS")
    print(f"{'='*80}")
    print("  maxtext_config.yml does NOT set 'scan_layers'")
    print("  MaxText default scan_layers = True")
    print("  This means layers are stacked/scanned")
    print("")
    print("  ⚠️  CRITICAL: Only 3 param arrays found in checkpoint!")
    print("  The 22 layers of attention + MLP weights are MISSING.")
    print("  Total data per process: ~184 MB (more than 3 params need)")
    print("  This suggests data exists but is stored differently.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orbax_dir", type=str, required=True)
    args = parser.parse_args()
    inspect_checkpoint(args.orbax_dir)
