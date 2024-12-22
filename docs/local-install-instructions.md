# Local install instructions

The course uses Python 3 and some data analysis packages such as numpy, pandas,
scikit-learn, and matplotlib.

## Install Miniconda

**This step is only necessary if you don't have conda installed already**:

- download the Miniconda installer for your operating system (Windows, MacOSX
  or Linux) [here](https://docs.conda.io/en/latest/miniconda.html)
- run the installer following the instructions
  [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)
  depending on your operating system.

## Create conda environment

```sh
# Clone this repo
git clone "git@github.com:HU-VisualAnalytics-GrE/Identify_Overall_Misclassification_Patterns.git"
cd Identify_Overall_Misclassification_Patterns/
# Create a conda environment with the required packages for this tutorial:
conda env create -f environment.yml
```

## Check your install

To make sure you have all the necessary packages installed, we **strongly
recommend** you to execute the `check_env.py` script located at the root of
this repository:

```sh
# Activate your conda environment
conda activate semester_project_group_e
python check_env.py
```

Make sure that there is no `FAIL` in the output when running the `check_env.py`
script, i.e. that its output looks similar to this:
