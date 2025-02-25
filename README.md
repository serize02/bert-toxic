# Toxic Bert

## Introduction

This project aims to develop a system that classifies messages from Dota 2 game chats based on their toxicity levels, focusing on insults and offensive comments. The core of the system is a neural network built upon BERT (Bidirectional Encoder Representations from Transformers), a language model developed by Hugging Face, implemented using PyTorch. The project also incorporates tools such as MLflow for experiment tracking, DagsHub, and DVC for data and version control, aligning with MLOps best practices.

[](./image.jpg)

## Local Usage

### Clone Repository

To obtain a local copy of the project, execute the following command:

```bash
git clone https://github.com/your_username/repository_name.git
```

### Setting Up the Python Environment

It is recommended to use a virtual environment to manage dependencies. Follow these steps:

```bash
python -m venv .env
source .env/bin/activate
```

### Install Dependencies

With the virtual environment activated, install the necessary dependencies:

```bash
pip install -r requiremtns.txt
```

## Pipeline

The project's workflow consists of the following stages:

*  Data Ingestion: Collect chat data from Dota 2 matches stored in Google Drive.
*  Data Split: Divide the collected data into training and testing datasets to facilitate model evaluation.
*  Setup Model: Define the primary architecture of the model, utilizing a pre-trained BERT model.
*  Train Model: Train the model using the training dataset, fine-tuning BERT to accurately classify the toxicity levels of chat messages.
*  Evaluate Model: Assess the model's performance on the test dataset using accuracy, f1_score (micro), f1_score (macro), confusion matrix and track all this on ML-Flow server on DagsHub

In order to execute the entire pipeline run:

```bash
dvc repro
```




