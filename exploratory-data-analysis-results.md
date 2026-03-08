# Exploratory Data Analysis (EDA) Results

## 1. Data Overview
* **Dimensions:** 970 rows, 11 columns, with 0 missing values.
* **Size Constraint:** With an 80/20 split, the training set is limited to ~776 samples, necessitating strict feature pruning to prevent overfitting.

## 2. Target Variable: `customer_sentiment`
* **Distribution:** Severe class imbalance exists. "Neutral" and "negative" sentiments dominate, while "positive" is a severe minority. 
* **Action:** Model evaluation will use the **F1-Macro Score**. The training loop will implement **Class-Weighted Cross-Entropy Loss**.

## 3. Text Feature: `conversation`
* **Sequence Lengths:** 53.30% of conversations exceed 350 words, guaranteeing many will break RoBERTa's 512-token limit.
* **Action:** A **Head + Tail Truncation** strategy will be used during tokenization to preserve both the context of the issue (head) and the resolution/final sentiment (tail).

## 4. Metadata Features & Selection Strategy
* **High-Cardinality (Dropped):** `issue_category` (40), `issue_sub_category` (109), `issue_category_sub_category` (109), and `product_sub_category` (50). These hold too many unique values for a small dataset and risk introducing noise.
* **Redundant (Dropped):** `agent_experience_level_desc` is a 1:1 duplicate of `agent_experience_level`.
* **Low-Cardinality (Retained):** `issue_area` (6), `product_category` (3), `issue_complexity` (3), and `agent_experience_level` (3). 
* **Integration Method:** The retained metadata will be injected into the model via **Text Prepending** (e.g., formatting them into a contextual string at the start of the `conversation` text) to leverage RoBERTa's native self-attention over text.
* 
