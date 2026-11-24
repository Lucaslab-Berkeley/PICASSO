"""Program for running CTF estimation on one image or a directory of micrographs."""

import time
from pathlib import Path

from picasso.managers import CTFEstimateManager

#######################################
### Editable parameters for program ###
#######################################

YAML_CONFIG_PATH = "/path/to/ctf_estimate_example_config.yaml"

##############################################################
### Main function called to run the CTF estimation program ###
##############################################################


def main() -> None:
    """Load YAML, estimate CTF for each image, and write Leopard/metrics outputs."""
    manager = CTFEstimateManager.from_yaml(YAML_CONFIG_PATH)
    n_images = len(manager.input_config.discover_images())
    counts = manager.worker_counts(n_images)
    print("Loaded configuration.")
    print(f"Using {len(counts)} device(s):")
    for device, k in counts:
        extra = ""
        if device.type == "cpu":
            extra = (
                f", T={manager.computational_config.cpu_threads_per_worker} "
                "threads/worker"
            )
        print(f"  {device}: K={k} concurrent image(s){extra}")
    print("Running CTF estimation...")

    start_time = time.time()
    results = manager.estimate_and_export()
    elapsed = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    output_dir = Path(manager.output_config.output_directory)
    print(f"Finished {len(results)} image(s).")
    print(f"CTF estimation wall time: {elapsed_str}")
    print(f"Optics CSV: {output_dir / manager.output_config.optics_csv_filename}")
    print(f"Metrics CSV: {output_dir / manager.output_config.metrics_csv_filename}")
    print("Per-micrograph optics YAML written next to those files.")
    print("Done!")


if __name__ == "__main__":
    main()
