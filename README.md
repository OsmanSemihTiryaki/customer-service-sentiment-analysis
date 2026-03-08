# Project Plan: Customer Support Sentiment Analysis (DI_725 Assignment 1)

## Objective
Classify the sentiment (positive, negative, neutral) of customer service conversations using an attention-based Transformer network.

## Phase 1: Exploratory Data Analysis (EDA) & Statistics
**Goal:** Understand the dataset's characteristics and uncover correlations that will drive feature selection.
* **Label Distribution:** Calculate the exact distribution of the `customer_sentiment` classes (positive, negative, neutral) to identify potential class imbalances.
* **Correlation Analysis:** Investigate relationships between metadata features (e.g., `issue_complexity`, `agent_experience_level`) and the target `customer_sentiment`.
* **Text Statistics:** Analyze the `conversation` lengths (token counts) to determine how many sequences exceed standard Transformer limits (e.g., 512 tokens).

## Phase 2: Feature Selection & Data Pre-processing
**Goal:** Clean the data and finalize the inputs for the model without leaking data from the test set.
* **Feature Pruning:** Based on EDA, drop uninformative or redundant columns (e.g., `agent_experience_level_desc` if redundant). 
* **Text Cleaning (Noise Removal):** Strip out non-semantic system logs (like "(After a few moments)") while preserving speaker tags ("Agent:", "Customer:") assuming token limits permit.
* **Data Splitting:** Create a strict, stratified Train/Validation split from `train.csv` to ensure the model generalizes well before final evaluation on `test.csv`.

## Phase 3: Tokenization & PyTorch Setup
**Goal:** Prepare the data for the specific architecture of the chosen Transformer model.
* **Tokenizer:** Implement Hugging Face's `RobertaTokenizer` (Byte-Pair Encoding) to map the text to `roberta-base` input IDs and attention masks.
* **Dataset/DataLoader:** Build custom PyTorch `Dataset` and `DataLoader` classes to handle batching efficiently.

## Phase 4: Model Training & Experiment Tracking
**Goal:** Fine-tune `roberta-base` and track performance rigorously.
* **Architecture:** Load `roberta-base` with a sequence classification head (3 output labels).
* **Tracking:** Initialize Weights and Biases (WANDB) to log training loss, validation loss, and hardware utilization in real-time.
* **Version Control:** Commit major milestones to a public GitHub repository logically and incrementally.

## Phase 5: Evaluation & IEEE Reporting
**Goal:** Assess the model and write a concise, professional report.
* **Metrics:** Evaluate the model on `test.csv` using appropriate metrics (Accuracy, F1-Score, Confusion Matrix) depending on class balance.
* **Documentation:** Draft a maximum 2-page report in IEEE format containing an Abstract, Introduction, Dataset EDA, Modeling choices, Evaluation, Results, and Discussion.
* **Deliverables Preparation:** Ensure the interactive `.ipynb` notebook is clean with outputs preserved, and package everything into `assignment_1_STUDENTNUMBER.zip`.
