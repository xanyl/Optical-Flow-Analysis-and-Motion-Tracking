import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("cv_project_output")

for name in ["sample1", "sample2"]:
    df = pd.read_csv(OUT_DIR / f"{name}_flow_stats.csv")

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_sec"], df["mean_magnitude"])
    plt.title(f"{name}: Mean Optical Flow Magnitude vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean flow magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_mean_flow_plot.png", dpi=200)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_sec"], df["moving_pixel_ratio_gt_0.5"])
    plt.title(f"{name}: Fraction of Moving Pixels vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Moving-pixel ratio (magnitude > 0.5)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_moving_ratio_plot.png", dpi=200)
    plt.show()

    max_row = df.loc[df["mean_magnitude"].idxmax()]
    print(f"\n{name}")
    print(f"  strongest motion around frame {int(max_row['frame_idx'])}")
    print(f"  strongest motion time = {max_row['time_sec']:.3f} sec")
    print(f"  strongest mean magnitude = {max_row['mean_magnitude']:.6f}")