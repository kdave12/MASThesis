# MAS Thesis Notebook
## Krishna Dave (UID: 905636874)

Privacy Auditing of Synthetic Data Using TAPAS Toolbox

This project contains a notebook with example code for privacy auditing using the TAPAS toolbox of synthetic data generated from the Breast Cancer Wisconsin Diagnostic Data (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). This toolbox is used for evaluating the privacy of synthetic data using adversarial techniques. The code for the TAPAS toolbox is from: https://github.com/alan-turing-institute/tapas 

## Setup Instructions
1. Install poetry (system-wide) from https://python-poetry.org/docs/ 
2. It can also be installed using pip: `pip install git+https://github.com/alan-turing-institute/privacy-sdg-toolbox`
3. Activate virtual environment inside the Jupyter notebook using `poetry shell`
4. Add the virtual environment to the available kernels for the notebook.
5. Make sure that this notebook is located in the correct directory inside the wider directory to ensure the correctness of the relative imports and file paths.

# GReaT Framework Installation
GReaT framework leverages the power of advanced pretrained Transformer language models to produce high-quality synthetic tabular data. Generate new data samples effortlessly with our user-friendly API in just a few lines of code. Please see our publication for more details: https://github.com/kathrinse/be_great

The GReaT framework can be easily installed using with pip - requires a Python version >= 3.9:
`pip install be-great`
