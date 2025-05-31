import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from glob import glob

def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard logs and save as CSV and plots."""
    print(f"\nSearching for TensorBoard logs in: {os.path.abspath(log_dir)}")
    
    # Find all event files
    event_paths = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_paths.append(os.path.join(root, file))

    if not event_paths:
        print(f"No TensorBoard event files found in {log_dir}")
        return

    print(f"Found {len(event_paths)} event files:")
    for path in event_paths:
        print(f"  - {path}")

    # Create output directories
    output_base = "tensorboard_data"
    os.makedirs(output_base, exist_ok=True)
    csv_dir = os.path.join(output_base, "csv")
    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nWill save exports to:")
    print(f"  CSV files: {os.path.abspath(csv_dir)}")
    print(f"  Plot images: {os.path.abspath(plots_dir)}")

    # Process each event file
    for event_path in event_paths:
        print(f"\nProcessing {event_path}")
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()

        # Get list of all scalar tags (metrics)
        tags = event_acc.Tags()["scalars"]
        print(f"Found {len(tags)} metrics:")
        for tag in tags:
            print(f"  - {tag}")
        
        # Extract data for each tag
        for tag in tags:
            print(f"\nExtracting {tag}")
            
            # Get all scalar events for this tag
            events = event_acc.Scalars(tag)
            
            # Convert to DataFrame
            data = pd.DataFrame(events)
            data.columns = ["wall_time", "step", "value"]
            
            # Save as CSV
            safe_tag = tag.replace("/", "_")
            csv_path = os.path.join(csv_dir, f"{safe_tag}.csv")
            data.to_csv(csv_path, index=False)
            print(f"Saved CSV to {csv_path}")
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(data["step"], data["value"])
            plt.title(tag)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(plots_dir, f"{safe_tag}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    # Search in all possible log directories
    log_dirs = [
        "./reinforcement_learning/runs",
        "./sac_ppo",
        "./ppo_simple_tb",
        "./her_sac_robot_tensorboard"
    ]
    
    print("Starting TensorBoard data export...")
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            extract_tensorboard_data(log_dir)
        else:
            print(f"\nDirectory not found: {log_dir}")
    
    print("\nExport complete! Check the tensorboard_data directory for all exports.") 