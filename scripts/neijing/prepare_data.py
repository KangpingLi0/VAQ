"""Resolve relocated image roots and validate every image without changing sources."""
import argparse
import hashlib
import json
from pathlib import Path

from run_experiment import DEFAULT_OUT, CHECKPOINT, save_json


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args()
    metadata = {}
    all_ids = {}
    for name, folder, image_directory in (
        ("calibration", "内镜报告生成校准数据", "images_224"),
        ("test", "内镜报告生成测试数据", "images"),
    ):
        root = Path("/data/lkp") / folder
        source = next(root.glob("*.json"))
        data = json.loads(source.read_text())
        missing = []
        seen = set()
        for row in data:
            if row["id"] in seen:
                raise ValueError(f"duplicate case {row['id']}")
            seen.add(row["id"])
            paths = [str(root / image_directory / Path(*Path(old).parts[-3:])) for old in row["images"]]
            missing.extend(path for path in paths if not Path(path).is_file())
            row["images"] = paths
            row["image"] = paths
            row["question"] = next(x["content"] for x in row["messages"] if x["role"] == "user").replace("<image>", "").strip()
            row["label"] = next(x["content"] for x in row["messages"] if x["role"] == "assistant")
        if missing:
            raise FileNotFoundError(missing[:10])
        save_json(args.output / (name + ".json"), data)
        all_ids[name] = seen
        metadata[name] = {"source": str(source), "source_sha256": sha256(source),
                          "cases": len(data), "images": sum(len(x["images"]) for x in data),
                          "missing_count": 0, "duplicate_ids": []}
    metadata["overlapping_case_ids"] = sorted(all_ids["calibration"] & all_ids["test"])
    assert not metadata["overlapping_case_ids"], "calibration/test leakage"
    save_json(args.output / "data_validation.json", metadata)
    inputs = list(Path("/data/lkp/eval").glob("*.py")) + list(Path("/data/lkp/eval").glob("*.json"))
    inputs += list(Path(CHECKPOINT).glob("*.json")) + list(Path(CHECKPOINT).glob("*.jinja"))
    if args.hash_weights:
        inputs += list(Path(CHECKPOINT).glob("*.safetensors"))
    save_json(args.output / "input_checksums.json", {str(path): sha256(path) for path in inputs})
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
