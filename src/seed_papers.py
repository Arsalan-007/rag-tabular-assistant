"""
Curated seed list of papers for the tabular-DL-vs-GBT research assistant.

Each entry is (arxiv_id, short_label). The arxiv_id is what the fetcher uses;
the label is only for human-readable logging.

Papers NOT on arXiv (LightGBM, CatBoost/Friedman journal versions, DeepGBM) are
listed separately in NON_ARXIV so the pipeline can report them as known gaps
rather than silently missing them. Download those manually and drop the PDFs in
data/pdfs/ using the given filename.
"""

# --- arXiv papers, grouped by research thread -----------------------------
SEED_PAPERS = [
    # 1. Core debate: trees vs deep learning on tabular
    ("2207.08815", "Grinsztajn - Why trees still outperform DL"),
    ("2106.03253", "Shwartz-Ziv - DL is Not All You Need"),
    ("2106.11959", "Gorishniy - Revisiting DL / FT-Transformer"),
    ("2106.11189", "Kadra - Well-tuned Simple Nets"),
    ("2407.00956", "Ye - A Closer Look at DL on Tabular"),
    ("2305.02997", "McElfresh - When Do NNs Outperform Boosted Trees"),

    # 2. Gradient boosting foundations
    ("1603.02754", "Chen & Guestrin - XGBoost"),
    ("1706.09516", "Prokhorenkova - CatBoost"),

    # 3. Attention / transformer architectures for tabular
    ("1908.07442", "Arik & Pfister - TabNet"),
    ("2012.06678", "Huang - TabTransformer"),
    ("2106.01342", "Somepalli - SAINT"),
    ("1810.11921", "Song - AutoInt"),
    ("2301.02819", "Chen - ExcelFormer"),
    ("2211.16887", "Yan - T2G-Former"),
    ("2106.02584", "Kossen - Self-Attention Between Datapoints (NPT)"),

    # 4. Boosting-inspired / tree-mimic neural nets
    ("2002.07971", "Badirli - GrowNet"),
    ("1909.06312", "Popov - NODE"),
    ("2002.07772", "Hazimeh - Tree Ensemble Layer"),
    ("2106.01613", "Chang - NODE-GAM"),
    ("1711.09784", "Frosst & Hinton - Soft Decision Tree"),

    # 5. Feature embeddings for tabular
    ("2203.05556", "Gorishniy - Embeddings for Numerical Features"),
    ("2410.24210", "Gorishniy - TabM"),

    # 6. Tabular foundation models / in-context
    ("2207.01848", "Hollmann - TabPFN"),
    ("2511.08667", "Hollmann - TabPFN-2.5"),

    # 7. Surveys (dense, high cross-reference value for RAG)
    ("2504.16109", "Survey - Representation Learning for Tabular Data"),
    ("2110.01889", "Borisov - DNNs and Tabular Data: A Survey"),
]

# --- Known-relevant papers NOT on arXiv (manual download required) ---------
# The pipeline logs these as gaps so you know exactly what the corpus is missing.
NON_ARXIV = [
    "LightGBM (Ke et al., NeurIPS 2017)",
    "Friedman - Greedy Function Approximation / GBM (Annals of Statistics 2001)",
    "DeepGBM (Ke et al., KDD 2019)",
]

# --- Optional keyword expansion to grow the corpus toward ~50 papers -------
# Set EXPANSION_TARGET to the total corpus size you want. The fetcher will
# search arXiv for these terms and add new papers until it hits the target.
EXPANSION_QUERIES = [
    "tabular deep learning",
    "attention tabular data",
    "gradient boosting neural network",
    "transformer tabular data",
]
EXPANSION_TARGET = 50  # set to len(SEED_PAPERS) to disable expansion
