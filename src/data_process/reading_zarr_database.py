import zarr
import numpy as np

# Set the path to the Zarr database
output_dir = "/home/yihan/Documents/acp_data/real_processed/flip_up_new_v5"  # Change this to your actual dataset path

# Open the Zarr database
buffer = zarr.open(output_dir, mode="r")

# Print the entire Zarr hierarchy
print("Zarr Database Structure:")
print(buffer.tree(expand=True))

# List all top-level groups
print("\nTop-level groups:", list(buffer.keys()))

# Inspect the metadata group if it exists
if "meta" in buffer:
    print("\nMetadata:")
    for key in buffer["meta"]:
        data = buffer["meta"][key]
        print(f"{key}: shape={data.shape}, dtype={data.dtype}")

# Inspect the "data" group which contains actual episodes
if "data" in buffer:
    print("\nEpisodes in Data Group:")
    for episode in buffer["data"]:
        ep_data = buffer["data"][episode]

        print(f"\nEpisode: {episode}")
        for dataset in ep_data:
            data_array = ep_data[dataset]
            print(f"  {dataset}: shape={data_array.shape}, dtype={data_array.dtype}")

        # Print the first few values of a sample dataset
        sample_key = list(ep_data.keys())[0]  # Choose the first dataset in the episode
        print(f"\nSample data from {sample_key}:")
        print(ep_data[sample_key][:5])  # Print the first 5 entries
