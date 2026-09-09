"""
Frozen, curated corpus for the tabular-DL-vs-GBT research assistant.

Each entry is (arxiv_id, short_label). The arxiv_id is what the fetcher uses;
the label is only for human-readable logging and corpus review.

WHY THIS LIST IS FROZEN
-----------------------
An earlier version of this project grew the corpus with arXiv keyword search
(queries like "tabular deep learning", "gradient boosting neural network").
That pulled in ~16 off-topic papers -- computational fluid dynamics, medical
imaging, "The Modern Mathematics of Deep Learning" -- which, because they are
long, accounted for more than half the chunks in the vector store. Retrieval
was searching mostly noise.

The corpus is now a hand-checked list: every paper below is about the core
question (why gradient-boosted trees do well on tabular data, and how neural /
attention architectures compare) or is a foundational reference for it. No
automatic expansion. To change the corpus, edit this file and re-run
`python src/ingest.py --reset`.

Papers NOT on arXiv (LightGBM, DeepGBM, Friedman's original GBM) are listed
separately in NON_ARXIV so the pipeline can report them as known gaps rather
than silently missing them.
"""

# --- Curated arXiv corpus, grouped by research thread ---------------------
SEED_PAPERS = [
    # 1. Core debate: trees vs deep learning on tabular
    ("2207.08815", "Grinsztajn - Why trees still outperform DL on typical tabular data"),
    ("2106.03253", "Shwartz-Ziv & Armon - Tabular Data: DL Is Not All You Need"),
    ("2305.02997", "McElfresh - When Do NNs Outperform Boosted Trees on Tabular Data"),
    ("2407.00956", "Ye - A Closer Look at Deep Learning Methods on Tabular Datasets"),
    ("2106.11189", "Kadra - Well-tuned Simple Nets Excel on Tabular Datasets"),
    ("2604.15297", "Benchmarking Optimizers for MLPs in Tabular Deep Learning"),

    # 2. Gradient boosting foundations
    ("1603.02754", "Chen & Guestrin - XGBoost: A Scalable Tree Boosting System"),
    ("1706.09516", "Prokhorenkova - CatBoost: unbiased boosting with categorical features"),

    # 3. Attention / transformer architectures for tabular
    ("1908.07442", "Arik & Pfister - TabNet: Attentive Interpretable Tabular Learning"),
    ("2012.06678", "Huang - TabTransformer: Contextual Embeddings for Categoricals"),
    ("2106.01342", "Somepalli - SAINT: Row Attention + Contrastive Pretraining"),
    ("1810.11921", "Song - AutoInt: Automatic Feature Interaction via Self-Attention"),
    ("2106.11959", "Gorishniy - Revisiting DL Models for Tabular Data (FT-Transformer)"),
    ("2211.16887", "Yan - T2G-Former: Tabular Features into Relation Graphs"),
    ("2106.02584", "Kossen - Self-Attention Between Datapoints (NPT)"),
    ("2301.02819", "Chen - ExcelFormer: a NN surpassing GBDTs on tabular data"),
    ("2305.18446", "Chen - Trompt: prompt-style deep network for tabular data"),
    ("2112.02962", "Chen - DANets: Deep Abstract Networks for Tabular Data"),

    # 4. Boosting-inspired / tree-mimic neural nets
    ("2002.07971", "Badirli - GrowNet: Gradient Boosting Neural Networks"),
    ("1909.06312", "Popov - Neural Oblivious Decision Ensembles (NODE)"),
    ("2002.07772", "Hazimeh - The Tree Ensemble Layer"),
    ("2106.01613", "Chang - NODE-GAM: Neural Generalized Additive Model"),
    ("1711.09784", "Frosst & Hinton - Distilling a NN Into a Soft Decision Tree"),
    ("2309.17130", "Marton - GRANDE: Gradient-Based Decision Tree Ensembles"),
    ("2006.06465", "Katzir - Net-DNF: A Neural Architecture for Tabular Data"),

    # 5. Feature embeddings for tabular
    ("2203.05556", "Gorishniy - On Embeddings for Numerical Features"),
    ("2410.24210", "Gorishniy - TabM: Parameter-Efficient Ensembling"),

    # 6. Tabular foundation models / in-context / LLMs
    ("2207.01848", "Hollmann - TabPFN: a Transformer for small tabular classification"),
    ("2511.08667", "Hollmann - TabPFN-2.5: advancing tabular foundation models"),
    ("2210.10723", "Hegselmann - TabLLM: Few-shot Tabular Classification with LLMs"),
    ("2305.06090", "Zhu - XTab: Cross-table Pretraining for Tabular Transformers"),
    ("2402.16785", "Kim - CARTE: Pretraining and Transfer for Tabular Learning"),

    # 7. Retrieval / nearest-neighbour tabular
    ("2307.14338", "Gorishniy - TabR: Tabular DL Meets Nearest Neighbors"),
    ("2407.03257", "Ye - Revisiting Nearest Neighbor for Tabular Data (ModernNCA)"),

    # 8. Transfer / few-shot / self-supervised
    ("2206.15306", "Levin - Transfer Learning with Deep Tabular Models"),
    ("2207.03208", "Rubachev - Revisiting Pretraining Objectives for Tabular DL"),
    ("2311.10051", "Tabular Few-Shot Generalization Across Heterogeneous Feature Spaces"),
    ("2402.01204", "Survey - Self-Supervised Learning for Non-Sequential Tabular Data"),

    # 9. Generative modelling of tabular data
    ("2209.15421", "Kotelnikov - TabDDPM: Modelling Tabular Data with Diffusion"),

    # 10. Surveys (dense, high cross-reference value for RAG)
    ("2110.01889", "Borisov - Deep Neural Networks and Tabular Data: A Survey"),
    ("2504.16109", "Survey - Representation Learning for Tabular Data"),
]

# --- Known-relevant papers NOT on arXiv (manual download required) ---------
# The pipeline logs these as gaps so you know exactly what the corpus is missing.
# To include one, drop its PDF in data/pdfs/ named <arxiv_id_or_slug>.pdf.
NON_ARXIV = [
    "LightGBM (Ke et al., NeurIPS 2017)",
    "Friedman - Greedy Function Approximation / GBM (Annals of Statistics 2001)",
    "DeepGBM (Ke et al., KDD 2019)",
]

# Sanity check: fail loudly on an accidental duplicate ID.
_ids = [pid for pid, _ in SEED_PAPERS]
assert len(_ids) == len(set(_ids)), (
    f"duplicate arXiv id in SEED_PAPERS: "
    f"{sorted({i for i in _ids if _ids.count(i) > 1})}"
)

CORPUS_SIZE = len(SEED_PAPERS)  # 41
