import os
import sys
import subprocess
import traceback

if __name__ == "__main__":
    print("SCRIPT STARTED")
    print()

    print()
    print("> echo $0")
    output_filename: str = "echo_0.txt"
    subprocess.run(f'echo $0 > {output_filename}', shell=True)
    with open(output_filename, "r") as file_handle:
        content = file_handle.read()
    print(content)

    print()
    print("> echo $1")
    output_filename: str = "echo_1.txt"
    subprocess.run(f'echo $1 > {output_filename}', shell=True)
    with open(output_filename, "r") as file_handle:
        content = file_handle.read()
    print(content)

    print()
    print("> which python")
    output_filename: str = "which_python.txt"
    subprocess.run(f'which python > {output_filename}', shell=True)
    with open(output_filename, "r") as file_handle:
        content = file_handle.read()
    print(content)

    print()
    print("> conda list")
    # help('modules')
    output_filename: str = "conda_list.txt"
    subprocess.run(f'conda list > {output_filename}', shell=True)
    with open(output_filename, "r") as file_handle:
        content = file_handle.read()
    print(content)

    print()
    print("> conda env list")
    output_filename: str = "conda_env_list.txt"
    subprocess.run(f'conda env list > {output_filename}', shell=True)
    with open(output_filename, "r") as file_handle:
        content = file_handle.read()
    print(content)

    print("Command Line Arguments:")
    print(sys.argv)
    print()

    print("Environment:")
    for key in os.environ:
        print(f"{key} = {os.environ[key]}")
    print()

    print()
    print("Current Working Directory:")
    print(os.getcwd())

    print()
    print("Files in Current Directory:")
    print(os.listdir())

    print()
    print("Adding current directory to PATH...")
    print("Current PATH:", sys.path)
    sys.path.insert(0, os.getcwd())
    print("Updated PATH:", sys.path)

    print()
    print("Data Folder")
    import argparse
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-d', '--data_folder')
    args = parser.parse_args()
    print(args.data_folder)
    print()
    print("Files in that Directory:")
    print(os.listdir(args.data_folder))
    print()
    pastis_datasets_yaml_path: str = os.path.join(args.data_folder, "PASTIS24", "datasets.yaml")
    print(f"pastis_datasets_yaml_path = '{pastis_datasets_yaml_path}'")
    print()
    print()
    print("Contents:")
    with open(pastis_datasets_yaml_path, "r") as file_handle:
        contents = file_handle.read()
    print(contents)
    print()
    print()
    print(f"Setting 'DATASET_INFO_PATH' environment variable to `{pastis_datasets_yaml_path}`")
    os.environ["DATASET_INFO_PATH"] = pastis_datasets_yaml_path
    print()
    print()
    print()
    print()

    print()
    print("> nvidia-smi")
    output_filename: str = "nvidia_smi.txt"
    subprocess.run(f'nvidia-smi > {output_filename}', shell=True)
    with open(output_filename, "r") as file_handle:
        content = file_handle.read()
    print(content)

    print()
    print("> nvcc --version")
    output_filename: str = "nvcc_version.txt"
    try:
        subprocess.run(f'nvcc --version > {output_filename}', shell=True)
        with open(output_filename, "r") as file_handle:
            content = file_handle.read()
        print(content)
    except Exception as exc:
        print(f"Exception raised: {exc}")

    print()
    print("torch.__version__")
    try:
        import torch
        print(torch.__version__)
    except Exception as exc:
        print(f"Exception raised: {exc}")
    print()

    print()
    print("torch.cuda.is_available()")
    try:
        import torch
        print(torch.cuda.is_available())
    except Exception as exc:
        print(f"Exception raised: {exc}")
    print()

    print()
    print("Importing Segmentation script...")
    from train_and_eval import segmentation_training_transf

    print()
    print("Running Segmentation script...")
    config_file: str = os.path.join(args.data_folder, "PASTIS24", "configs", "TSViT-S_fold1.yaml")
    device_ids: list = [0]
    base_dir: str = os.path.join(args.data_folder, "PASTIS24")
    try:
        segmentation_training_transf.run(config_file=config_file, device_ids=device_ids, base_dir=base_dir)
    except Exception as exc:
        print(f"Exception raised: {exc}")
        print(traceback.format_exc())

    print()
    print("Done.")
