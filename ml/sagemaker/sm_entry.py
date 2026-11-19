"""SageMaker script-mode entry point (XGBoost 1.7-1 framework container).

Runs exactly the same ml/train_core.py as local training. The source tarball built by
ml/train.py contains this file at its root plus the ml/ package. Keep this code compatible
with the container's Python (3.8+): no match statements, no zip(strict=...).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xgboost as xgb  # noqa: E402

from ml.train_core import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_round", type=int, default=300)
    ap.add_argument("--max_train_rows", type=int, default=1_500_000)
    ap.add_argument("--horizons", type=str, default="15,30,60")
    ap.add_argument("--max_depth", type=int, default=7)
    ap.add_argument("--eta", type=float, default=0.1)
    args, _ = ap.parse_known_args()
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    res = train(os.path.join(train_dir, "training.npz"), model_dir,
                horizons_min=tuple(int(h) for h in args.horizons.split(",")),
                num_round=args.num_round, max_train_rows=args.max_train_rows,
                params={"max_depth": args.max_depth, "eta": args.eta},
                trained_on="SageMaker training job (ml.m5.large)")
    # The built-in serving stack loads a file named "xgboost-model": save it in native format
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, "model.json"))
    booster.save_model(os.path.join(model_dir, "xgboost-model"))
    print(json.dumps({h: per["xgboost"] for h, per in res["horizons"].items()}))


if __name__ == "__main__":
    main()
