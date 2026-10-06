# Lab: Data Preparation, EDA, Preprocessing & Distributed Processing

This lab has four parts. Part 1 loads a dataset and splits it into training and validation sets. Part 2 explores the data to check it is good enough for predicting a project's tag from its title and description. Part 3 preprocesses the data (cleaning, encoding, tokenizing) so it is ready for model training. Part 4 runs the same data processing in a distributed way with Ray so it can scale beyond a single machine.

**Author:** Your Name
**Course:** Your Course Name
**Date:** Your Date

---

## Objectives

- Ingest data from a CSV file into a Pandas DataFrame
- Understand the purpose of the train, validation and test splits
- Create a stratified split with scikit-learn and verify it
- Analyze the class (tag) distribution and spot data imbalance
- Use word clouds to check whether text features carry signal for each class
- Clean text data, encode labels, and tokenize text with a pretrained tokenizer
- Wrap all preprocessing into one reusable function
- Scale ingestion, splitting and preprocessing across machines with Ray Data

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- wordcloud
- nltk
- transformers
- ray[data]

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk transformers "ray[data]"
```

## Dataset

A CSV of machine learning projects, loaded from the Made With ML repository:

```
https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv
```

| Column | Description |
|--------|-------------|
| `id` | Unique project ID |
| `created_on` | Creation timestamp |
| `title` | Project title |
| `description` | Project description |
| `tag` | Class label (one per project) |

Classes: `natural-language-processing`, `computer-vision`, `other`, `mlops`

---

## Part 1: Data Preparation

### 1.1 Ingestion

```python
import pandas as pd

DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
df = pd.read_csv(DATASET_LOC)
df.head()
```

### 1.2 Class counts

```python
df.tag.value_counts()
```

| Tag | Count |
|-----|-------|
| natural-language-processing | 310 |
| computer-vision | 285 |
| other | 106 |
| mlops | 63 |

### 1.3 Stratified split

Each project has exactly one tag (multi-class task), so the split is stratified on `tag` to keep similar class proportions in both sets.

```python
from sklearn.model_selection import train_test_split

test_size = 0.2
train_df, val_df = train_test_split(
    df, stratify=df.tag, test_size=test_size, random_state=1234
)
```

### 1.4 Validate the split

```python
# Train counts
train_df.tag.value_counts()

# Validation counts, scaled up to be comparable with train
val_df.tag.value_counts() * int((1 - test_size) / test_size)
```

| Tag | Train | Validation (scaled) |
|-----|-------|---------------------|
| natural-language-processing | 248 | 248 |
| computer-vision | 228 | 228 |
| other | 85 | 84 |
| mlops | 50 | 52 |

The counts are very close, so the class distribution is preserved.

### Key concepts

- **Train split:** used to fit the model's weights (inputs and labels).
- **Validation split:** used after each epoch to evaluate performance and tune hyperparameters such as the learning rate.
- **Test split:** a separate holdout set used once after training to estimate performance on unseen data.

> **Tip:** Keep the test set separate from the training data. If the test set is re-split every time the training data grows, models become hard to compare.

---

## Part 2: Exploratory Data Analysis (EDA)

### What is EDA?

- It is not just producing a fixed set of plots (e.g., a correlation matrix).
- The goal is to convince yourself the data is sufficient for the task.
- Use it to answer specific questions and make insights easier to find.
- It is not a one-time step: revisit it as data grows to catch distribution shifts and anomalies.

### 2.1 Setup

```python
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import warnings; warnings.filterwarnings("ignore")
from wordcloud import WordCloud, STOPWORDS
```

### 2.2 Tag distribution

Question: how many projects do we have per tag?

```python
all_tags = Counter(df.tag)
all_tags.most_common()

tags, tag_counts = zip(*all_tags.most_common())
plt.figure(figsize=(10, 3))
ax = sns.barplot(x=list(tags), y=list(tag_counts))
ax.set_xticklabels(tags, rotation=0, fontsize=8)
plt.title("Tag distribution", fontsize=14)
plt.ylabel("# of projects", fontsize=12)
plt.show()
```

**Finding:** There is some class imbalance, but it is moderate. If needed, it can be handled by over-sampling rare classes, under-sampling common ones, or using class weights in the loss function.

### 2.3 Word clouds

Question: do the title and description contain signal that is unique to each tag?

```python
tag = "natural-language-processing"
plt.figure(figsize=(10, 3))
subset = df[df.tag == tag]
text = subset.title.values
cloud = WordCloud(
    stopwords=STOPWORDS, background_color="black", collocations=False,
    width=500, height=300).generate(" ".join(text))
plt.axis("off")
plt.imshow(cloud)
plt.show()
```

Change `tag` to view other classes, and repeat with `description` instead of `title`.

**Finding:** The most frequent words per tag match intuition, so the title and description are useful features for modeling.

---

## Part 3: Data Preprocessing

Preprocessing has two kinds of steps: **preparing** (organizing and cleaning) and **transforming** (encoding and engineering features).

> **Warning:** Some steps are *global* (e.g., lower-casing, removing stop words) and some are *local* (learned from the training split only, e.g., vocabulary or standardization). For local steps, split the data first to avoid data leakage.

### Overview of common techniques

**Preparing**

| Technique | Idea |
|-----------|------|
| Joins | Combine tables into one view; use point-in-time valid joins to avoid leaks |
| Missing values | Drop rows, drop the feature, or fill values (e.g., with the mean); missing values may appear as `0`, `null`, `NA` |
| Outliers | Define what is "normal" (e.g., within 2 standard deviations); be careful not to remove important outliers such as fraud |
| Feature engineering | Combine features to draw out signal, ideally with domain experts |
| Cleaning | Apply constraints, keep data types consistent; for images crop/resize, for text lower/stem/lemmatize/regex |

**Transforming**

| Technique | Idea |
|-----------|------|
| Scaling | Standardization (mean 0, std 1), min-max (0 to 1), binning (continuous to categorical); learn from train split only |
| Encoding | Label (index), one-hot (binary vector), embeddings (dense vectors that capture context) |
| Extraction | PCA, n-gram counts, autoencoders, transfer learning |

### 3.1 Setup

```python
import json
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
```

### 3.2 Feature engineering

Title and description are combined into a single input feature.

```python
df["text"] = df.title + " " + df.description
```

### 3.3 Cleaning

```python
nltk.download("stopwords")
STOPWORDS = stopwords.words("english")

def clean_text(text, stopwords=STOPWORDS):
    """Clean raw text string."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  # remove links

    return text
```

Apply it to the DataFrame and compare before and after:

```python
original_df = df.copy()
df.text = df.text.apply(clean_text)
print(f"{original_df.text.values[0]}\n{df.text.values[0]}")
```

Then drop unused columns and rows with a null tag:

```python
df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")
df = df.dropna(subset=["tag"])
df = df[["text", "tag"]]
df.head()
```

> **Note:** Emojis and punctuation can carry signal, but it is best to start with a simple feature set and add more later while checking whether they help.

### 3.4 Label encoding

Class labels are converted to indices using the **training split** only.

```python
tags = train_df.tag.unique().tolist()
num_classes = len(tags)
class_to_index = {tag: i for i, tag in enumerate(tags)}
class_to_index

df["tag"] = df["tag"].map(class_to_index)
```

Decode predictions back to text labels:

```python
def decode(indices, index_to_class):
    return [index_to_class[index] for index in indices]

index_to_class = {v: k for k, v in class_to_index.items()}
decode(df.head()["tag"].values, index_to_class=index_to_class)
```

### 3.5 Tokenization

The text is tokenized with the SciBERT tokenizer (the same model that is fine-tuned later). It returns **token ids** and an **attention mask** (1 for real tokens, 0 for padding).

```python
from transformers import BertTokenizer

def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))

tokenize(df.head(1))
```

`padding="longest"` pads every sequence in a batch to the length of the longest one so inputs have a uniform size.

### 3.6 Putting it together

All steps are wrapped in one function so the same preprocessing can be reused for training, validation, and inference.

```python
def preprocess(df, class_to_index):
    """Preprocess the data."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # clean dataframe
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs

preprocess(df=train_df, class_to_index=class_to_index)
```

> **Note:** Run `preprocess` on the original split DataFrames (`train_df`, `val_df`), which still contain the `title` and `description` columns.

---

## Part 4: Distributed Data Processing

So far everything ran on one machine in a single Python process. If the dataset no longer fits in memory (large unstructured data, LLMs), the processing must be distributed across machines.

[Ray](https://www.ray.io/) is used here because it scales Python code with minimal changes, and it integrates well with later workloads (training, tuning, serving) and with other frameworks such as Dask, Modin and Spark. The dataset in this lab is small on purpose, but the same code works on a much larger dataset, and adding compute makes it faster with no code changes.

### 4.1 Setup

Ray is set to preserve order so results are reproducible and deterministic.

```python
import ray

ray.data.DatasetContext.get_current().execution_options.preserve_order = True  # deterministic
```

### 4.2 Ingestion

```python
ds = ray.data.read_csv(DATASET_LOC)
ds = ds.random_shuffle(seed=1234)
ds.take(1)
```

### 4.3 Splitting

Ray has a built-in `train_test_split`, but a modified version (`stratify_split`) is used so the split is stratified on `tag`. It comes from the course repository (`madewithml/data.py`), so clone that repo or copy the function into your project.

```python
import sys
sys.path.append("..")
from madewithml.data import stratify_split

test_size = 0.2
train_ds, val_ds = stratify_split(ds, stratify="tag", test_size=test_size)
```

### 4.4 Distributed preprocessing

The Pandas-based `preprocess` function from Part 3 is reused without changes. `map_batches` applies it across batches of the data in a distributed manner.

```python
tags = train_ds.unique(column="tag")
class_to_index = {tag: i for i, tag in enumerate(tags)}

sample_ds = train_ds.map_batches(
    preprocess,
    fn_kwargs={"class_to_index": class_to_index},
    batch_format="pandas")
sample_ds.show(1)
```

Each output row contains `ids` (token ids), `masks` (attention mask) and `targets` (encoded label).

---

## Conclusions

- The stratified split keeps class proportions similar in train and validation.
- The class imbalance is mild and manageable.
- Text features contain meaningful signal for predicting tags.
- Text is cleaned, labels are encoded, and inputs are tokenized into ids and attention masks, ready for model training.
- With Ray Data, the same preprocessing function scales to larger datasets and more compute without code changes.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. Install the requirements (see above).
3. Open the notebook or run the script:
   ```bash
   jupyter notebook lab.ipynb
   # or
   python lab.py
   ```

## Reference

Based on the *Preparation*, *Exploration*, *Preprocessing* and *Distributed* lessons from [Made With ML](https://madewithml.com/) by Goku Mohandas.
