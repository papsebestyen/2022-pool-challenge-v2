from pathlib import Path
import json

import boto3
import requests


bucket = "borza-public-data"
subdir = "swimming-pool-packs"

DEFAULT_PACK = "ch2-S-50"
region = "eu-central-1"


def load_pack(pack_id=DEFAULT_PACK, out_path="./pack.zip"):
    bpath = f"{subdir}/{pack_id}.zip"
    if pack_id == DEFAULT_PACK:
        r = requests.get(f"https://{bucket}.s3.{region}.amazonaws.com/{bpath}")
        Path(out_path).write_bytes(r.content)
        return 

    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, bpath, out_path)


def evaluate(solution_dir, true_results):
    out = json.loads((Path(solution_dir) / "output.json").read_text())
    if out == true_results:
        return True

    msg = f"len(true_results)={len(true_results)}, len(output)={len(out)}"
    for i, (res, calc) in enumerate(zip(true_results, out)):
        if res != calc:
            msg += f"\nTRUE RESULT != OUTPUT at {i}:\n{res} != {calc}"
            break
    print(f"-------" * 10, "EVALUATION ERROR", msg, f"-------" * 10, sep="\n")
    return False
