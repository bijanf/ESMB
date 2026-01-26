
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

def verify_biharmonic(run_name, label, file_index=-1):
    output_dir = f"outputs/{run_name}"
    files = sorted(glob.glob(os.path.join(output_dir, "mean_*.nc")))
    
    if not files:
        print(f"No files found in {output_dir}")
        return

    # Select file
    if file_index == -1:
        f = files[-1]
    else:
        # Check bounds
        if file_index < len(files):
            f = files[file_index]
        else:
             print(f"Index {file_index} out of range for {len(files)} files.")
             return

    print(f"Analyzing {f}...")
    ds = xr.open_dataset(f, decode_times=False)

    # Extract Fields
    sst = ds.ocean_temp.isel(z=0).values - 273.15
    sss = ds.ocean_salt.isel(z=0).values
    
    # Calculate SSS Laplacian (Noise Metric)
    sss_roll_xm = np.roll(sss, 1, axis=1)
    sss_roll_xp = np.roll(sss, -1, axis=1)
    sss_roll_ym = np.roll(sss, 1, axis=0)
    sss_roll_yp = np.roll(sss, -1, axis=0)
    laplacian_sss = np.abs(sss_roll_xp + sss_roll_xm + sss_roll_yp + sss_roll_ym - 4*sss)
    
    # Analyze SST Gradients (Central Diff)
    # Gx = (T(x+1) - T(x-1)) / 2dx
    # Gy = ...
    # Just use raw difference magnitude
    sst_grad_x = np.abs(sst - np.roll(sst, 1, axis=1))
    sst_grad_y = np.abs(sst - np.roll(sst, 1, axis=0))
    sst_grad_mag = np.sqrt(sst_grad_x**2 + sst_grad_y**2)
    
    # Metrics
    valid_sss = sss > 10.0
    if np.sum(valid_sss) > 0:
        noise_mean = np.mean(laplacian_sss[valid_sss])
        noise_max = np.max(laplacian_sss[valid_sss])
        print(f"[{label}] SSS Noise Mean: {noise_mean:.4f} PSU")
        print(f"[{label}] SSS Noise Max:  {noise_max:.4f} PSU")
    
    print(f"[{label}] SST Mean Gradient Mag: {np.mean(sst_grad_mag):.4f} K/grid")
    
    # Boundary Check
    print(f"[{label}] SST North Boundary Mean: {np.mean(sst[-1, :]):.4f} C")
    print(f"[{label}] SST South Boundary Mean: {np.mean(sst[0, :]):.4f} C")
    print(f"[{label}] SSS North Boundary Mean: {np.mean(sss[-1, :]):.4f} PSU")
    print(f"[{label}] SSS South Boundary Mean: {np.mean(sss[0, :]):.4f} PSU")
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # SST
    im1 = axes[0].imshow(sst, origin='lower', cmap='RdBu_r', vmin=-2, vmax=30)
    axes[0].set_title(f'{label}: SST (C)')
    plt.colorbar(im1, ax=axes[0])
    
    # SSS
    im2 = axes[1].imshow(sss, origin='lower', cmap='viridis', vmin=32, vmax=38)
    axes[1].set_title(f'{label}: SSS (PSU)')
    plt.colorbar(im2, ax=axes[1])
    
    # SSS Noise
    im3 = axes[2].imshow(laplacian_sss, origin='lower', cmap='inferno', vmin=0, vmax=5)
    axes[2].set_title(f'{label}: SSS Noise (Laplacian)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plot_filename = f"verify_{run_name}.png"
    plt.savefig(plot_filename)
    print(f"Saved {plot_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="control_run_prod_v4_test_bi_1e15")
    parser.add_argument("--label", type=str, default="Biharmonic Test")
    parser.add_argument("--index", type=int, default=-1, help="File index (0 for first)")
    args = parser.parse_args()
    
    verify_biharmonic(args.run_dir, args.label, args.index)
