"""
Labeled evaluation set for the tabular-DL RAG assistant.

Each entry pairs a natural-language question with the arXiv ID(s) of the
paper(s) that specifically answer it -- the ground truth for retrieval scoring.
Retrieval is credited if it surfaces ANY listed paper (paper-level judgement,
not chunk-level).

Curation rules used here:
  * The question must have a single clearly-correct source paper in the corpus
    (or a small set, when a topic is genuinely introduced by more than one).
  * Questions are phrased the way a researcher would actually ask -- not by
    pasting the paper's title.
  * Every ID below is present in src/seed_papers.py (the frozen corpus).

`OUT_OF_DOMAIN` holds questions the corpus should NOT answer well; the eval
uses them to check that the low-confidence guard fires (best dense similarity
below settings.min_similarity) instead of confidently returning noise.
"""

EVAL_QUESTIONS = [
    # ── Core debate: trees vs deep learning ────────────────────────────
    {
        "q": "Why do tree-based models still outperform deep learning on tabular data?",
        "papers": ["2207.08815"],
        "note": "Grinsztajn: NNs are biased toward smooth solutions, hurt by uninformative features, and not rotation-invariant.",
    },
    {
        "q": "What inductive biases make tree ensembles well-suited to irregular, non-smooth target functions?",
        "papers": ["2207.08815"],
    },
    {
        "q": "Is deep learning all you need for tabular data, or do gradient-boosted trees still win on typical benchmarks?",
        "papers": ["2106.03253"],
        "note": "Shwartz-Ziv & Armon.",
    },
    {
        "q": "Under what dataset conditions do neural networks actually beat gradient-boosted trees on tabular data?",
        "papers": ["2305.02997"],
        "note": "McElfresh: NN advantage correlates with dataset 'irregularity' metrics.",
    },
    {
        "q": "How well do simple, well-regularised MLPs perform on tabular benchmarks when their regularisation is tuned?",
        "papers": ["2106.11189"],
        "note": "Kadra - a cocktail of 13 regularisers.",
    },
    {
        "q": "Which optimizer choices matter most when training plain MLPs on tabular data?",
        "papers": ["2604.15297"],
    },
    {
        "q": "What are the main families of deep learning architectures proposed for tabular data?",
        "papers": ["2110.01889", "2504.16109", "2407.00956"],
        "note": "Any of the two surveys or Ye's benchmarking study.",
    },

    # ── Gradient boosting foundations ─────────────────────────────────
    {
        "q": "How does XGBoost handle sparse features and scale split-finding efficiently?",
        "papers": ["1603.02754"],
        "note": "Sparsity-aware split finding + weighted quantile sketch.",
    },
    {
        "q": "How does CatBoost reduce target leakage / prediction shift when encoding categorical features?",
        "papers": ["1706.09516"],
        "note": "Ordered target statistics + ordered boosting.",
    },

    # ── Attention / transformer architectures ─────────────────────────
    {
        "q": "How does the FT-Transformer turn numerical and categorical features into tokens for a transformer?",
        "papers": ["2106.11959"],
        "note": "Feature Tokenizer introduced in Gorishniy 'Revisiting DL Models'.",
    },
    {
        "q": "How does TabNet use sequential attention to pick which features to reason about at each decision step?",
        "papers": ["1908.07442"],
    },
    {
        "q": "How does SAINT combine row (inter-sample) attention with column attention, and what pretraining does it use?",
        "papers": ["2106.01342"],
    },
    {
        "q": "How does TabTransformer build contextual embeddings for categorical columns?",
        "papers": ["2012.06678"],
    },
    {
        "q": "How does AutoInt model high-order feature interactions with multi-head self-attention?",
        "papers": ["1810.11921"],
    },
    {
        "q": "How can attention operate between different data points instead of only within a single row's features?",
        "papers": ["2106.02584", "2106.01342"],
        "note": "NPT is the canonical answer; SAINT's inter-sample attention also qualifies.",
    },
    {
        "q": "How does T2G-Former organise tabular features into relation graphs to promote heterogeneous interaction?",
        "papers": ["2211.16887"],
    },
    {
        "q": "How does ExcelFormer use attention plus its own data augmentation to surpass GBDTs on tabular data?",
        "papers": ["2301.02819"],
    },
    {
        "q": "How does Trompt derive per-sample feature importances using learned prompt tokens?",
        "papers": ["2305.18446"],
    },
    {
        "q": "How do DANets group correlated raw features into abstract features with a learnable sparse mask?",
        "papers": ["2112.02962"],
    },

    # ── Boosting-inspired / tree-mimic neural nets ────────────────────
    {
        "q": "What makes GrowNet a boosting-inspired neural architecture rather than a single deep network?",
        "papers": ["2002.07971"],
        "note": "Gradient boosting of weak shallow-MLP learners with a corrective step.",
    },
    {
        "q": "How do Neural Oblivious Decision Ensembles (NODE) make tree-style splits differentiable?",
        "papers": ["1909.06312"],
        "note": "entmax feature selection + soft oblivious decision trees.",
    },
    {
        "q": "How can a decision-tree ensemble be embedded as a differentiable neural network layer with conditional computation?",
        "papers": ["2002.07772", "1909.06312", "2309.17130"],
        "note": "Tree Ensemble Layer is the most literal; NODE and GRANDE also qualify.",
    },
    {
        "q": "How does GRANDE learn axis-aligned decision-tree ensembles end-to-end with gradient descent?",
        "papers": ["2309.17130"],
    },
    {
        "q": "How can a trained neural network be distilled into a soft binary decision tree for interpretability?",
        "papers": ["1711.09784"],
        "note": "Frosst & Hinton.",
    },
    {
        "q": "How does NODE-GAM stay interpretable as a generalized additive model while using neural oblivious trees?",
        "papers": ["2106.01613"],
    },
    {
        "q": "How does Net-DNF bake a disjunctive-normal-form (AND/OR) inductive bias into a network for tabular data?",
        "papers": ["2006.06465"],
    },

    # ── Feature embeddings ───────────────────────────────────────────
    {
        "q": "Why do learned embeddings for continuous numerical features improve tabular deep models, and what forms work best?",
        "papers": ["2203.05556"],
        "note": "Piecewise-linear and periodic (Fourier-feature) embeddings.",
    },
    {
        "q": "How does TabM get an implicit ensemble out of a single MLP with parameter sharing (BatchEnsemble-style)?",
        "papers": ["2410.24210"],
    },

    # ── Foundation models / in-context / LLMs ───────────────────────
    {
        "q": "How does TabPFN classify a small tabular dataset in a single forward pass without task-specific training?",
        "papers": ["2207.01848"],
        "note": "Prior-data fitted network; Bayesian inference amortised into a transformer.",
    },
    {
        "q": "How does TabPFN-2.5 push in-context tabular learning to larger datasets than the original TabPFN?",
        "papers": ["2511.08667"],
    },
    {
        "q": "How does TabLLM serialise a table row into natural language so a large language model can classify it few-shot?",
        "papers": ["2210.10723"],
    },
    {
        "q": "How does XTab pretrain one shared transformer backbone across tables that have different columns?",
        "papers": ["2305.06090"],
        "note": "Dataset-specific featurizers + shared backbone, federated pretraining.",
    },
    {
        "q": "How does CARTE represent a table row as a graph so it can transfer across schemas without matching columns?",
        "papers": ["2402.16785"],
    },

    # ── Retrieval / nearest-neighbour tabular ───────────────────────
    {
        "q": "How does TabR add a retrieval (nearest-neighbour) component to a feed-forward tabular network?",
        "papers": ["2307.14338"],
    },
    {
        "q": "How does ModernNCA revive neighbourhood component analysis as a competitive deep tabular baseline?",
        "papers": ["2407.03257"],
    },

    # ── Transfer / self-supervised ─────────────────────────────────
    {
        "q": "Which self-supervised pretraining objectives actually help deep tabular models downstream?",
        "papers": ["2207.03208", "2402.01204"],
        "note": "Rubachev's study; the SSL survey also catalogues them.",
    },
    {
        "q": "How can deep tabular models transfer learned representations to a new dataset with a different feature set?",
        "papers": ["2206.15306", "2305.06090", "2402.16785"],
    },
    {
        "q": "How can a tabular model generalise few-shot to datasets whose feature spaces do not overlap at all?",
        "papers": ["2311.10051"],
    },

    # ── Generative ────────────────────────────────────────────────
    {
        "q": "How does TabDDPM apply diffusion models to mixed continuous and categorical tabular data?",
        "papers": ["2209.15421"],
        "note": "Gaussian diffusion for numerical + multinomial diffusion for categorical.",
    },
]

# Questions the corpus should NOT be able to answer well. Used to check that the
# low-confidence guard fires instead of confidently returning irrelevant chunks.
OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "How do I fix a leaking kitchen tap?",
    "Summarise the plot of Hamlet.",
    "What were the main causes of the 2008 financial crisis?",
]
