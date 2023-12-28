# MAS Thesis Notebook
## Krishna Dave (UID: 905636874)

Privacy Auditing of Synthetic Data Using TAPAS Toolbox

This project contains a notebook with example code for privacy auditing using the TAPAS toolbox of synthetic data generated from the Breast Cancer Wisconsin Diagnostic Data (https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). This toolbox is used for evaluating the privacy of synthetic data using adversarial techniques. The code for the TAPAS toolbox is from: https://github.com/alan-turing-institute/tapas 

## Setup Instructions

### Poetry Installation
1. Install poetry (system-wide) from https://python-poetry.org/docs/ 
2. It can also be installed using pip: `pip install git+https://github.com/alan-turing-institute/privacy-sdg-toolbox`
3. Activate virtual environment inside the Jupyter notebook using `poetry shell`
4. Add the virtual environment to the available kernels for the notebook.
5. Make sure that this notebook is located in the correct directory inside the wider directory to ensure the correctness of the relative imports and file paths.

### GReaT Framework Installation
GReaT framework leverages the power of advanced pretrained Transformer language models to produce high-quality synthetic tabular data. Generate new data samples with their API in just a few lines of code. Checkout their publication for more details: https://github.com/kathrinse/be_great

The GReaT framework can be easily installed using with pip - requires a Python version >= 3.9:
`pip install be-great`

Colab Example (Run on a GPU): https://colab.research.google.com/github/kathrinse/be_great/blob/main/examples/GReaT_colab_example.ipynb

#### Debugging Tips/Links
LLM model in the GReaT API has to be run on a GPU, it doesn't work on a Mac CPU. 

Otherwise, you will encounter the following errors: 
`RuntimeError: CUDA Out of memory` 
`RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1`

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

<img src="/gpt4.png" alt="Alternative text" />


