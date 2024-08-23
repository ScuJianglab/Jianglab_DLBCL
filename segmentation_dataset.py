import os
import shutil
from sklearn.model_selection import train_test_split

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def split_dataset(folder_A, folder_B, output_dir, test_size=0.1, val_size=0.2):
    # Ensure output directories exist
    create_dir_if_not_exists(output_dir)
    train_A_dir = os.path.join(output_dir, 'trainA')
    train_B_dir = os.path.join(output_dir, 'trainB')
    val_A_dir = os.path.join(output_dir, 'valA')
    val_B_dir = os.path.join(output_dir, 'valB')
    test_A_dir = os.path.join(output_dir, 'testA')
    test_B_dir = os.path.join(output_dir, 'testB')

    create_dir_if_not_exists(train_A_dir)
    create_dir_if_not_exists(train_B_dir)
    create_dir_if_not_exists(val_A_dir)
    create_dir_if_not_exists(val_B_dir)
    create_dir_if_not_exists(test_A_dir)
    create_dir_if_not_exists(test_B_dir)

    # Get list of files in folders
    files_A = sorted(os.listdir(folder_A))
    files_B = sorted(os.listdir(folder_B))

    # Ensure that both folders have the same files
    assert files_A == files_B, "The files in both folders do not match."

    # Split the filenames into training+validation and test sets
    train_val_files, test_files = train_test_split(files_A, test_size=test_size, random_state=42)

    # Split the training+validation set into training and validation sets
    train_files, val_files = train_test_split(train_val_files, test_size=val_size, random_state=42)

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(folder_A, file), os.path.join(train_A_dir, file))
        shutil.copy(os.path.join(folder_B, file), os.path.join(train_B_dir, file))

    for file in val_files:
        shutil.copy(os.path.join(folder_A, file), os.path.join(val_A_dir, file))
        shutil.copy(os.path.join(folder_B, file), os.path.join(val_B_dir, file))

    for file in test_files:
        shutil.copy(os.path.join(folder_A, file), os.path.join(test_A_dir, file))
        shutil.copy(os.path.join(folder_B, file), os.path.join(test_B_dir, file))

    print(f"Train, validation, and test sets created successfully in {output_dir}")





# Example usage
folder_A = '/homellm8t/zhaoxz/DAPI/masked_patch/patches/IHC_DAPI'
folder_B = '/homellm8t/zhaoxz/DAPI/masked_patch/patches/IHC'
output_dir = '/homellm8t/zhaoxz/Path_wsi_dataset/TrainValAB'

split_dataset(folder_A, folder_B, output_dir)
