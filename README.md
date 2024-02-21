# Text Summarizer FYP

## Table of Contents
- [Objective](#objective)
- [Tasks Completed](#tasks-completed)
- [Key Technologies Used](#key-technologies-used)
- [Demo](#Demo)

## Objective
Develop a transcription and text summarizer for multi-speaker videos, providing users with key insights and topics discussed in webinars and panel discussions.

## Tasks Completed
1. **Data Collection**
   - Utilized TIB, QMSUM, and BBC News datasets for evaluating model performance.
2. **Model Evaluation**
   - Assessed OpenAI and GCP VertexAI models, comparing strengths and weaknesses.
3. **Summarization Techniques Exploration**
   - Explored TextRank and MapReduce for summarization.
   - Tested prompt engineering, temperature adjustment, few-shot prompting, and retrieval augmented generation.
4. **Web Application Development**
   - Created a Streamlit web app for interactive use.

## Key Technologies Used
- **OpenAI**: Model customization and text summarization.
- **GCP VertexAI**: Model evaluation.
- **Streamlit**: Web application development.

## Demo
Youtube Link: https://www.youtube.com/watch?v=X1ZRpSjR5z0

This repository contains all code to process, evaluate and fine tune models. Might be abit messy, most important evaluation file is All Model File with .ipynb and the most important automation and evaluation metrics code is in /helper.

```bash
# create virtual environment
py -m venv ".venv"
cd .venv/Scripts
activate.bat # for windows
source .venv/Scripts/activate # for linux

# install relevant libraries
pip install -r requirements.txt

```

# or install manually

```bash
pip install pandas langchain selenium pyperclip
pip install torch torchtext datasets
pip install py-readability-metrics rouge rouge_score nltk absl-py bert-score
python -m nltk.downloader punkt
....
```
