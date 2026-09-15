"""Export raw social events into upstream DyGLib's processed-data format."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from interactiondynamics.data.social import SocialConfig, SocialEventDataset


def prepare(name, root, target):
    ds = SocialEventDataset(SocialConfig(name=name, root=root))
    directory = Path(target) / "processed_data" / name
    directory.mkdir(parents=True, exist_ok=False)
    count = len(ds.src)
    # Preserve homogeneous IDs and raw event ordering; zero is reserved padding.
    pd.DataFrame(dict(u=ds.src.numpy() + 1, i=ds.dst.numpy() + 1,
                      ts=(ds.timestamps - ds.timestamps[0]).astype(np.float64),
                      label=np.ones(count), idx=np.arange(1, count + 1))).to_csv(directory / f"ml_{name}.csv", index=False)
    features = np.ones((count + 1, 1), dtype=np.float32)
    features[0] = 0
    np.save(directory / f"ml_{name}.npy", features)
    np.save(directory / f"ml_{name}_node.npy", np.zeros((len(ds.node_ids) + 1, 1), dtype=np.float32))
    print(f"Exported {count} events to {directory}; no temporal binning applied")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="college_msg", choices=["college_msg", "email_eu_core", "sociopatterns"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--target", default="derived/dyglib")
    args = parser.parse_args()
    prepare(args.dataset, args.root, args.target)
