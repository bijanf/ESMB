
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def verify_seam(run_name, label, file_index=-1):
    output_dir = f"outputs/{run_name}"
    files = sorted(glob.glob(os.path.join(output_dir, "mean_*.nc")))
    
    if not files:
        print(f"No files found in {output_dir}")
        return

    if file_index == -1:
        f = files[-1]
    else:
        f = files[file_index]
        
    print(f"Analyzing {f}...")
    ds = xr.open_dataset(f, decode_times=False)

    sst = ds.ocean_temp.isel(z=0).values - 273.15
    mask = ds.ocean_temp.isel(z=0).values != 0 # Approx mask
    
    # Check Seam: Column 0 vs Column -1
    col_0 = sst[:, 0]
    col_end = sst[:, -1]
    diff = np.abs(col_0 - col_end)
    
    print(f"[{label}] Seam Max Diff: {np.max(diff):.4f} C")
    print(f"[{label}] Seam Mean Diff: {np.mean(diff):.4f} C")
    
    # Plot Seam profile at equator
    ny = sst.shape[0]
    equator_idx = ny // 2
    
    # Check Atmos Seam
    if 'atmos_temp' in ds:
        atmos_t = ds.atmos_temp.isel(lev=0).values if 'lev' in ds.dims else ds.atmos_temp.values
        diff_atm = np.abs(atmos_t[:, 0] - atmos_t[:, -1])
        print(f"[{label}] Atmos Temp Seam Max Diff: {np.max(diff_atm):.4f} K")
        
    # Check Flux Seam
    if 'net_heat_flux' in ds:
        flux = ds.net_heat_flux.values
        diff_flux = np.abs(flux[:, 0] - flux[:, -1])
        print(f"[{label}] Heat Flux Seam Max Diff: {np.max(diff_flux):.4f} W/m2")

    # Check Velocity Seam
    if 'ocean_u' in ds:
        u = ds.ocean_u.isel(z=0).values
        diff_u = np.abs(u[:, 0] - u[:, -1])
        print(f"[{label}] Velocity (U) Seam Max Diff: {np.max(diff_u):.4f} m/s")

    # Check Polar Fluxes
    if 'net_heat_flux' in ds:
        # Flux shape (ny, nx)
        flux_N = np.mean(flux[-1, :])
        flux_S = np.mean(flux[0, :])
        print(f"[{label}] Heat Flux North Mean: {flux_N:.4f} W/m2")
        print(f"[{label}] Heat Flux South Mean: {flux_S:.4f} W/m2")
        # Check if Flux is zero (indicating masking issue)
        if flux_N == 0.0: print(f"[{label}] WARNING: North Flux is Exact Zero!")
        if flux_S == 0.0: print(f"[{label}] WARNING: South Flux is Exact Zero!")


    # Check Mask Seam
    # Infer mask from sst being 0 ideally, or use loaded mask if available?
    # We'll rely on SST=0 check roughly
    mask = sst != 0
    # If mask differs across seam, that's a topological wall
    mask_diff = mask[:, 0] != mask[:, -1]
    if np.any(mask_diff):
        print(f"[{label}] WARNING: Land Mask Mismatch at Seam! {np.sum(mask_diff)} points.")
        # Print locations
        rows = np.where(mask_diff)[0]
        print(f"[{label}] Mismatch Rows: {rows}")

    # Atmospheric Polar Check
    if 'atmos_temp' in ds:
        atmos_t = ds.atmos_temp.isel(lev=0).values if 'lev' in ds.dims else ds.atmos_temp.values
        # Shape (y_atm, x_atm)
        atm_N = np.mean(atmos_t[-1, :])
        atm_S = np.mean(atmos_t[0, :])
        print(f"[{label}] Atmos Temp North Mean: {atm_N:.2f} K ({atm_N-273.15:.2f} C)")
        print(f"[{label}] Atmos Temp South Mean: {atm_S:.2f} K ({atm_S-273.15:.2f} C)")
        
    # Plot Seam profile at equator
    ny = sst.shape[0]
    equator_idx = ny // 2
    
    # Plot Meridional Profile (Zonal Mean)
    sst_zonal_mean = np.mean(sst, axis=1) # (ny,)
    lat = np.linspace(-90, 90, ny)

    plt.figure(figsize=(10, 15))
    
    plt.subplot(3,1,1)
    plt.plot(sst[equator_idx, :], label="SST at Equator")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.axvline(x=sst.shape[1]-1, color='r', linestyle='--')
    plt.title(f"{label}: SST Zonal Profile (Equator)")
    plt.legend()
    
    plt.subplot(3,1,2)
    plt.plot(lat, sst_zonal_mean, label="SST Zonal Mean")
    plt.xlabel("Latitude")
    plt.ylabel("SST (C)")
    plt.title(f"{label}: Meridional SST Profile")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3,1,3)
    if 'net_heat_flux' in ds:
        plt.plot(flux[equator_idx, :], label="Heat Flux")
        plt.title(f"{label}: Heat Flux Zonal Profile (Equator)")
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(f"seam_profile_{run_name}.png")
    print(f"Saved seam_profile_{run_name}.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="control_run_prod_v4")
    parser.add_argument("--label", type=str, default="Prod V4")
    parser.add_argument("--index", type=int, default=-1)
    args = parser.parse_args()
    
    verify_seam(args.run_dir, args.label, args.index)
