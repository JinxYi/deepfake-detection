
import sys
import os

import sys
import os


########### COPY AND PASTE THIS SECTION TO THE START OF EVERY NOTEBOOK ###########
########### START SECTION ###########
def is_colab_env():
    return "google.colab" in sys.modules

def mount_google_drive(drive_dir="/content/drive/", repo_dir="MyDrive/repositories/deepfake-detection"):
    # mount google drive
    from google.colab import drive
    drive.mount(drive_dir)

    # change to correct working directory
    import os
    os.chdir(f"{drive_dir}{repo_dir}")
    print(os.listdir()) # verify content

def resolve_path(levels_deep=3):
    if is_colab_env():
        mount_google_drive()
    else:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath('__file__'))

        # Construct the path to the parent directory
        for i in range(levels_deep):
            current_dir = os.path.dirname(current_dir)

        # Add the parent directory to sys.path
        sys.path.append(current_dir)
        print(sys.path)
########### END SECTION ###########

def make_directory(base_dir, new_dir):
    """Creates a directory given a base directory. No error is raised if the directory already exists."""
    new_path = os.path.join(base_dir, new_dir)
    os.makedirs(new_path, exist_ok=True)
    return new_path
