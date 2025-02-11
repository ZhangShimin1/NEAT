import itertools
import os
import yaml
import subprocess

neuron_type = 'ltc'
# Define parameter ranges for grid search
param_grid = {
    'decay': [0.3, 0.5, 0.7],
    'threshold': [0.3, 0.5, 0.7],
    'beta': [0.05, 0.1, 0.15],
    'alpha': [0.4, 0.6, 0.8]
}

# Load template config
with open('conf/LTC_template.yaml', 'r') as f:
    template = f.read()

# Generate all combinations of parameters
keys = param_grid.keys()
combinations = list(itertools.product(*[param_grid[key] for key in keys]))

# Read the original run.sh content once before the loop
with open('run.sh', 'r') as f:
    original_run_content = f.read()

# Create configs and run experiments for each combination
for i, values in enumerate(combinations):
    # Create parameter dictionary
    params = dict(zip(keys, values))
    
    # Create config name
    config_name = f"{neuron_type}_grid_{i}"
    
    # Create config content by replacing placeholders
    config_content = template
    for key, value in params.items():
        config_content = config_content.replace(key.upper(), str(value))
    
    # Write config file
    config_path = f"conf/{config_name}.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # Update run.sh with new config name and exp name
    new_run_content = original_run_content.replace(
        'default_config_name="grid_search"',
        f'default_config_name="{config_name}"'
    ).replace(
        f'exp_name="grid_search"',
        f'exp_name="{config_name}"'
    )
    
    with open('run.sh', 'w') as f:
        f.write(new_run_content)
    
    # Run the experiment
    print(f"\nRunning experiment {i+1}/{len(combinations)}")
    print(f"Parameters: {params}")
    subprocess.run(['bash', 'run.sh'])

# Restore original run.sh
with open('run.sh', 'w') as f:
    f.write(original_run_content) 
    