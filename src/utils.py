
import sys
import os
import config

from models.resnet import ResNetClassifier


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

def get_model(model_tag: str = "dwt_rgb_resnet18"):
    in_channels = 1
    model_checkpoint_dir = f"{sys.path[0]}/{config.CHECKPOINTS_DIR}/{model_tag}"
    
    match model_tag:
        case "dct_resnet18":
            in_channels = 1
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=8-step=228240.ckpt"
        case "dwt_resnet18":
            in_channels = 4
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=15-step=405760.ckpt"
        case "dwt_rgb_resnet18":
            in_channels = 7
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=13-step=177520.ckpt"
        case "fft_magnitude_phase_resnet18":
            in_channels = 2
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=8-step=228240.ckpt"
        case "fft_magnitude_resnet18":
            in_channels = 1
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=10-step=278960.ckpt"
        case "fft_phase_resnet18":
            in_channels = 1
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=6-step=177520.ckpt"
        case "rgb_resnet18":
            in_channels = 3
            checkpoint = "lightning_logs/version_0/checkpoints/epoch=10-step=278960.ckpt"
    
    deepfake_detector = ResNetClassifier.load_from_checkpoint(
        checkpoint_path=f"{model_checkpoint_dir}/{checkpoint}",
        in_channels=in_channels,
        freeze_features=False,
        weights_only=False
        )
    return deepfake_detector