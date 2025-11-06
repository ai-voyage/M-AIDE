# M-AIDE
The official repository for M-AIDE: Mechanistic Agentic Interpretability for Decoding Empathy in Language Models - IVCNZ2025

## Release
- [6/11/2024] Installation guide and requirements added. The code for training EFD agent and interpretability agent will be released soon.

## Installation
To set up the repository on your local machine:

```
1. Clone the repository:
```
   git clone https://github.com/ai-voyage/M-AIDE.git
   cd M-AIDE
```
2. Create a virtual environment and install dependencies
```
  conda create -n your_env_name python=3.10
  conda activate your_env_name
```
3. Install dependencies
```
  pip install -r requirements.txt

## Usage
1. Data Preparation:
   -Download the dataset available at: [E-THER: A Multimodal Dataset for Empathic AI -- Towards Emotional Mismatch Awareness](https://arxiv.org/abs/2509.02100)
   -Download the therapeutic agent from: [E-THER: A Multimodal Dataset for Empathic AI -- Towards Emotional Mismatch Awareness](https://arxiv.org/abs/2509.02100)
2. Extract Activations:
   - Run the extraction script:
```bash
exract_activations.py.py
 ```
We will use these activations to train the EFD agent.
