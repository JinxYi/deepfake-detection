## Environment Setup

### Install Environment

To install the conda environment with the necessary dependencies installed, enter the following in the command prompt:
```
source .bashrc # activate conda bash
conda env create -f environment.yml
```

### Activating environment

To activate an existing conda environment, enter the following into the command prompt:
```
source activate
conda activate mlenv
```

To check your conda environments, enter: 
```
source env list
```

To run the Flask server, enter:
```
conda run --no-capture-output -n df-env python server.py
```
where `df-env` is the name of the conda environment installed earlier. Add the `--no-capture-output` to see print statements in the console.