import yaml
import subprocess

# Path to the YAML file and script
yaml_file = "393.yaml"
script_path = "prepare_dataset.py"

# Function to load the YAML configuration
def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

# Function to save the updated YAML configuration
def save_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

# Function to run the dataset preparation
def run_prepare_dataset(training_views):
    for view in training_views:
        # Load the YAML configuration
        config = load_yaml(yaml_file)

        # Validate the structure of the YAML file
        if "dataset" not in config or "subject" not in config["dataset"]:
            raise ValueError("The 'subject' key is missing in the 'dataset' section of the YAML file.")
        if "output" not in config or "dir" not in config["output"] or "name" not in config["output"]:
            raise ValueError("The 'output' section with 'dir' and 'name' keys is missing in the YAML file.")

        # Update the training_view and resolve placeholders
        config["training_view"] = view
        config["output"]["dir"] = config["output"]["dir"].format(
            subject=config["dataset"]["subject"]
        )
        config["output"]["name"] = f"cam_{view}"  # Dynamically update the name for each training view

        # Debugging - Print the updated configuration
        print(f"Updated config for training_view={view}:\n{config}")

        # Save the updated YAML file
        save_yaml(config, yaml_file)

        # Run the prepare_dataset script
        command = f"python3 {script_path} --cfg {yaml_file}"
        print(f"Running command: {command}")
        result = subprocess.run(command, shell=True)

        # Check if the script ran successfully
        if result.returncode != 0:
            print(f"Error occurred while processing training_view: {view}")
            break

if __name__ == "__main__":
    training_views = range(1, 23)  # Training views 1 to 22
    run_prepare_dataset(training_views)
