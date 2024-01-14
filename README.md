# MASDS Thesis Notebook
## Krishna Dave (UID: 905636874)
## Adversarial Privacy Auditing of Synthetically Generated Data produced by Large Language Models using the TAPAS Toolbox

This  repository contains example implementations of privacy auditing using the TAPAS toolbox for synthetic data generated from [The Breast Cancer Wisconsin Diagnostic Data](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). This toolbox is used for evaluating the privacy of synthetic data using adversarial techniques. The code for the TAPAS toolbox is from: https://github.com/alan-turing-institute/tapas ([Associated Paper](https://arxiv.org/abs/2211.06550))

Here is the example notebook exemplifying privacy attacks, training and testing threat models and generation of privacy auditing metrics and reports for Breast Cancer Synthetic Data using the TAPAS toolbox: https://github.com/kdave12/MASThesis/blob/main/ExampleNotebooks/PrivacyAuditing_SyntheticData_Tapas.ipynb

## Breast Cancer Dataset

The real dataset from [The Breast Cancer Wisconsin Diagnostic Data](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) is used to generate synthetic data.

For the dataset, the features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. Ten real-valued features computed for each cell nucleus are as follows:
- radius (mean of distances from the center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter² / area — 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension (“coastline approximation” — 1)

This database is also available through the UW CS ftp server:
- `ftp ftp.cs.wisc.edu`
- `cd math-prog/cpo-dataset/machine-learn/WDBC/`

## Exploratory Data Analysis (EDA) on Real and Synthetic Datasets

Breast cancer is a malignant type of cancer and has been life threatening for women around the world. With early detection as a non-metastatic disease, breast cancer is curable. Healthcare datasets such as these are highly confidential, but important for analysis and model training that help with early detection. Therefore, exploring the use cases of synthetic data generation for such a dataset is quite beneficial.

Exploratory data analysis (EDA) is conducted on the real dataset and the synthetically generated datasets from several sources (including GReaT, OpenAI's GPT4, Mostly.AI). The purpose is to understand the data quality and data distributions, and compare and contrast between the real dataset and the synthetically generated ones. EDA includes checking for data types, data previews, missing values, constant occurences, duplicate rows, univariate analysis, bivariate analysis and multivariate analysis.

An open-source Python library called `Edvart` is used to explore datasets and generate EDA reports. `Edvart` is available on PyPI and can be installed using `pip` as such: 
`pip install edvart`

- Alternatively, the library can be downloaded from: https://github.com/datamole-ai/edvart
- More info about the library can be found here: https://datamole-ai.github.io/edvart/
- The notebooks with EDA reports for real and synthetic datasets are found here: https://github.com/kdave12/MASThesis/tree/main/ExampleNotebooks
- The HTML files with interactive data visualizations for the EDA reports can be found here: https://github.com/kdave12/MASThesis/tree/main/HTML_files_EDA

## TAPAS Toolbox and Framework Env - Setup Instructions

To mimic the TAPAS environment exactly, `poetry` is recommended.

### Poetry Installation
1. Install poetry (system-wide) from https://python-poetry.org/docs/ 
2. Installation using pip: `pip install git+https://github.com/alan-turing-institute/privacy-sdg-toolbox`
3. Activate virtual environment inside the Jupyter notebook using `poetry shell`
4. Add the virtual environment to the available kernels for the notebook.
5. Make sure that this notebook is located in the correct directory inside the wider directory to ensure the correctness of the relative imports and file paths.

## Synthetic Data Generators - Setup Instructions

### GReaT Framework Installation
GReaT framework leverages the power of advanced pretrained Transformer language models to produce high-quality synthetic tabular data. We can generate new data samples with their API. Checkout the following for more details: https://github.com/kathrinse/be_great ([Associated Paper](https://openreview.net/forum?id=cEygmQNOeI))

The GReaT framework can be installed using pip - requires a Python version >= 3.9:
`pip install be-great`

Example Notebook for synthetic data generation using the BeGReaT framework: https://github.com/kdave12/MASThesis/blob/main/ExampleNotebooks/BreastCancer_BeGreat_SyntheticDataGeneration.ipynb

Colab Example (To run on a GPU): https://colab.research.google.com/github/kathrinse/be_great/blob/main/examples/GReaT_colab_example.ipynb

#### Debugging Tips/Links
LLM model in the GReaT API has to be run on a GPU, it doesn't work on a Mac CPU. 

Otherwise, will encounter the following errors: 
`RuntimeError: CUDA Out of memory` 
`RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1`
`RuntimeError: CUDA out of memory. Tried to allocate 12.50 MiB (GPU 0; 10.92 GiB total capacity; 8.57 MiB already allocated; 9.28 GiB free; 4.68 MiB cached) `

To solve this, run this on Google Colab, navigate to Edit > Notebook settings > select Runtime type to be Python 3 > Hardware accelerator to T4 GPU or paid A100 GPU/V100 GPU.

For the error: 
`An error has occurred: Breaking the generation loop!`
`To address this issue, consider fine-tuning the GReaT model for an longer period. This can be achieved by increasing the number of epochs.
Alternatively, you might consider increasing the max_length parameter within the sample function. For example: model.sample(n_samples=10, max_length=2000)
If the problem persists despite these adjustments, feel free to raise an issue on our GitHub page at: https://github.com/kathrinse/be_great/issues`

To solve this error, try passing in `fp16=true` in the model definition, increasing the `max_length` value and increasing the number of `epochs`.

Here are some links that helped me with debugging: 
- https://stackoverflow.com/questions/68166721/pytorch-fails-with-cuda-error-device-side-assert-triggered-on-colab
- https://discuss.pytorch.org/t/cuda-error-device-side-assert-triggered-cuda-kernel-errors-might-be-asynchronously-reported-at-some-other-api-call-so-the-stacktrace-below-might-be-incorrect-for-debugging-consider-passing-cuda-launch-blocking-1/160825/5
- https://github.com/pytorch/pytorch/issues/75534
- https://stackoverflow.com/questions/70340812/how-to-install-pytorch-with-cuda-support-with-pip-in-visual-studio
- https://medium.com/@snk.nitin/how-to-solve-cuda-out-of-memory-error-850bb247cfb2
- https://stackoverflow.com/questions/64589421/packagesnotfounderror-cudatoolkit-11-1-0-when-installing-pytorch
- https://github.com/pytorch/pytorch/issues/30664
- https://github.com/kathrinse/be_great/issues/42
- https://github.com/kathrinse/be_great/issues/40
- https://research.google.com/colaboratory/faq.html#gpu-availability
- https://github.com/pytorch/pytorch/issues/16417
- https://github.com/googlecolab/colabtools/issues/3409

### Using OpenAI's GPT4 to generate synthetic data

The OpenAI Python library provides convenient access to the OpenAI REST API from any Python 3.7+ application. The API can be installed using `pip install openai`.
The documentation for OpenAI Python API library can be found here: https://github.com/openai/openai-python

We can use OpenAI's GPT4 to generate synthetic data from real dataset using the API or chat with a prompt such as this with an attachment of the real dataset:
"Can you create synthetic data that mimics this real dataset, and output a csv file with that data."

#### Useful Links
- https://github.com/openai/openai-cookbook
- https://github.com/openai/openai-python
- https://www.reddit.com/r/OpenAI/comments/161cygf/gpt4_api_access/?rdt=53543
- https://platform.openai.com/docs/introduction
- https://platform.openai.com/playground
- https://platform.openai.com/docs/api-reference
- https://platform.openai.com/api-keys
- https://platform.openai.com/docs/api-reference/authentication
- https://community.openai.com
- https://platform.openai.com/usage

ChatGPT4 Example:

<img src="/images/gpt4.png" alt="Alternative text" />

### Mostly.AI's AI-generated synthetic data - Free Trial

Mostly AI (https://mostly.ai) is a synthetic data generator startup that provides AI-powered synthetic data generation for free of charge for up to 100K rows daily.

Here is the documentation for preparing the data: https://mostly.ai/docs/guides/prepare-data

Other links:
- https://mostly.ai/synthetic-data-dictionary
- https://mostly.ai/synthetic-data/what-is-synthetic-data
- https://mostly.ai/synthetic-data-platform/synthetic-data-generation
- https://mostly.ai/blog/how-to-generate-synthetic-data

<img src="/images/mostlyai2.png" alt="Alternative text" />
<img src="/images/mostlyai.png" alt="Alternative text" />
