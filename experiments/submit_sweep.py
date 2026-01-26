
import subprocess
import os
import time

def submit_sweep():
    """
    Submits a batch of control runs with different parameters.
    """
    
    # Sweep Parameters
    # r_drag: Friction. Lower = Stronger AMOC, Higher = Weaker? 
    # Previous run with 7.9e-4 was too strong (16000 Sv?? Wait, 16000 was extremely high).
    # Typical r_drag ~ 1e-2 to 1e-1 ??
    # Default was 5e-2. 
    # Tuned 7.9e-4 might have been too small (too slippery -> massive flow).
    # Let's sweep around a reasonable range: [1e-3, 5e-3, 1e-2, 2e-2]
    
    r_drags = [0.001, 0.005, 0.01, 0.02]
    
    # kappa_gm: Eddy Diffusivity. 
    # Higher = More mixing, generally flattens isopycnals -> Weaker AMOC?
    # Default 1000.
    
    kappas = [500.0, 1000.0, 2000.0]
    
    # Total jobs: 4 * 3 = 12 jobs.
    
    # Ensure generated scripts directory exists
    os.makedirs("experiments/generated", exist_ok=True)
    
    for r in r_drags:
        for k in kappas:
            job_name = f"sweep_r{r}_k{int(k)}"
            suffix = f"sweep_r{r}_k{int(k)}"
            
            print(f"Submitting {job_name}...")
            
            # Create a specific slurm script for this run
            # This is robust.
            
            script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.log
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Load necessary modules
module load anaconda
module load cuda/12.3.1

# Activate Virtual Environment
source venv/bin/activate

# Set Resolution
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH
export JAX_ENABLE_X64=True

echo "Starting Sweep Run: r_drag={r}, kappa_gm={k}"
echo "Job ID: $SLURM_JOB_ID"

# Run the simulation
# Run for 5 years to see stable AMOC tendencies
stdbuf -o0 -e0 python experiments/control_run.py --years 5.0 --r_drag {r} --kappa_gm {k} --suffix {suffix}

echo "Run finished"
"""
            script_path = f"experiments/generated/run_{job_name}.sh"
            with open(script_path, "w") as f:
                f.write(script_content)
                
            # Submit
            subprocess.run(["sbatch", script_path])
            
            # Sleep slightly to avoid spamming scheduler
            time.sleep(1.0)
            
if __name__ == "__main__":
    submit_sweep()
