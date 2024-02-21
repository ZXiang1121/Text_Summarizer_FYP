# Text Summarizer for Webinars

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
