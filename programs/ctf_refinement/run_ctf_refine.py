"""Program for running CTF refinement using 2D template matching."""

import pickle
import time
from pathlib import Path

from picasso.managers import CTFRefineManager

#######################################
### Editable parameters for program ###
#######################################

# Edit your YAML file to configure the CTF refinement program.
# See online documentation for more information on editing this file.
YAML_CONFIG_PATH = "/path/to/ctf_refine_example_config.yaml"

# Path where the refined CTF parameters will be output as a pickle file.
# If None, the output_directory from the config will be used.
OUTPUT_PATH = None  # Set to a specific path, or None to use config output_directory

##############################################################
### Main function called to run the CTF refinement program ###
##############################################################


def main() -> None:
    """Main function for running the CTF refinement program."""
    ctf_manager = CTFRefineManager.from_yaml(YAML_CONFIG_PATH)

    print("Loaded configuration.")
    print("Running CTF refinement...")

    start_time = time.time()

    refined_ctf_params = ctf_manager.refine_ctf()

    print("Finished core CTF refinement call.")

    # Print the wall time of the refinement in HH:MM:SS
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"CTF Refinement wall time: {elapsed_time_str}")

    # Exporting the refined CTF parameters to a pickle file
    print("Exporting results...")

    if OUTPUT_PATH is not None:
        # Use the specified output path
        output_path = Path(OUTPUT_PATH)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(refined_ctf_params, f)
        print(f"Refined CTF parameters saved to: {output_path}")
    else:
        # Use the manager's write method with config output_directory
        ctf_manager.write_refined_ctf_params(refined_ctf_params)
        if ctf_manager.output_config.output_directory is not None:
            output_path = (
                Path(ctf_manager.output_config.output_directory)
                / "refined_ctf_params.pkl"
            )
            print(f"Refined CTF parameters saved to: {output_path}")
        else:
            print("Warning: No output directory specified. Results not saved to disk.")
            print("Refined CTF parameters are available in the returned dictionary.")

    print("Done!")


# NOTE: Invoking program under `if __name__ == "__main__"` necessary for multiprocessing
if __name__ == "__main__":
    main()
