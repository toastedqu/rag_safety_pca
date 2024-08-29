# Simple RAG Safety Detection with PCA

Disclaimer: This project digressed into OOD (out-of-domain) query detection instead. However, as long as you collect actual negative examples, there is no need to modify any part except data processing.

Datasets (16):
- Internal: COVID, Substance
- MSMARCO: biomedical, music, film, finance, law, computing
- Stackoverflow: history, crypto, chess, cooking, astronomy, fitness, anime, literature

Methods (6):
- E-family: E-ball, E-cube (same edge), E-cube (different edges)
- Linear: LogReg, SVM
- Clustering: GMM 

Ablation Study (based on digressed topic): 
- 3 different PC projections using different bi-encoders (NO statistical significance via 2-sample Kolmogorov-Smirnov test)
    - MPNet (main)
    - DistilBERT
    - DistilROBERTa
- 2 different feature computations (NO visible difference in performance)
    - Projection of Query embeddings on Doc PCs
    - Distance between Query embeddings and Doc PCs
- Whether to sort PCs based on individual performance (NO visible difference in performance, which does NOT make sense)
    - Sort PC by explained variance (i.e., default order)
    - Sort PC by individual performace on dataset
- Performance vs #PCs
- Performance vs Radius (E-family only)
- Optimal hyperparameters per dataset (E-family only)

## How to run

The main file is `main.py`. You can run various entrypoints by choosing the appropriate arguments.
We suggest the following order of commands:

### Generate embeddings 
```bash
python main.py generate_embeddings 
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to use for embeddings. Default: `all-mpnet-base-v2`.
- `--dataset`: Dataset to generate embeddings for. Default: `stackexchange`.
- `--tags`: Tags to filter the dataset. Default: `all`.

### Generate PCA
```bash
python main.py generate_pcas
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to use for embeddings. Default: `all-mpnet-base-v2`.
- `--tags`: Tags to filter the dataset. Default: `all`.

### Generate Datasets
```bash
python main.py generate_datasets
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to use for embeddings. Default: `all-mpnet-base-v2`.
- `--tags`: Tags to filter the dataset. Default: `all`.
- `--negative_tags`: Negative tags to filter the dataset. Default: `all`.
- `--metric`: Metric to use. Default: `proj` (projection).
- `--test_size`: Test size for train-test split. Default: `0.2`.


### Search for best hyperparameters
```bash
python main.py search_hyperparameters
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to use for embeddings. Default: `all-mpnet-base-v2`.
- `--radiuses`: Radiuses to search for. Default: `[0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]`.
- `--methods`: Methods to search for. Default: `all` (equals to ['eball', 'ecube', 'erect', 'logreg', 'svm', 'gmm']).
- `--tags`: Tags to filter the dataset. Default: `all`.
- `--metric`: Metric to use. Default: `proj` (projection).

This command will store the result in the cache folder as a txt file for a method and a tag.

### Train and evaluate a model
```bash
python main.py train_and_evaluate
```

Parameters to set:
- `--seed`: Seed for reproducibility. Default: `2`.
- `--model_name`: Model to use for embeddings. Default: `all-mpnet-base-v2`.
- `--methods`: Methods to train and evaluate. Default: `all` (equals to ['eball', 'ecube', 'erect', 'logreg', 'svm', 'gmm']).
- `--tags`: Tags to filter the dataset. Default: `all`.
- `--metric`: Metric to use. Default: `proj` (projection).
- `--radius`: Radius to use when method is eball or ecube. Default: `0.01`.
- `--length`: Length to use when method is erect. Default: `0.01`.
- `--width`: Width to use when method is erect. Default: `0.01`.
- `--n_pcas`: Number of PCs to use. Default: `5`.