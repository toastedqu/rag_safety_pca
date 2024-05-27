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