"""
Labeled evaluation set for the tabular-DL RAG assistant.

Each entry pairs a natural-language question with the arXiv ID(s) of the paper(s)
that specifically answer it -- the ground truth for retrieval scoring. A question
may list more than one relevant paper when the topic is genuinely covered by
several (retrieval is credited if it finds ANY of them).

>>> REVIEW THIS FILE. <<<
These labels were drafted from each paper's distinctive contribution, but you know
the literature -- fix any mapping that's wrong, delete questions that are ambiguous,
and add your own. The eval is only as trustworthy as these labels. Aim for ~20 you
fully believe in.

Only IDs present in your corpus (src/seed_papers.py) will score; anything else is
reported as "label not in corpus" so you can catch typos.
"""

EVAL_QUESTIONS = [
    # --- Core debate: trees vs deep learning ---
    {
        "q": "Why do tree-based models still outperform deep learning on tabular data?",
        "papers": ["2207.08815"],  # Grinsztajn
        "note": "Grinsztajn's central finding: MLP-like models struggle with irregular targets, uninformative features, rotation non-invariance.",
    },
    {
        "q": "What inductive biases make tree ensembles well-suited to irregular target functions?",
        "papers": ["2207.08815"],
    },
    {
        "q": "Is deep learning all you need for tabular data, or do gradient-boosted trees still win?",
        "papers": ["2106.03253"],  # Shwartz-Ziv & Armon
    },
    {
        "q": "When do neural networks outperform boosted trees on tabular datasets?",
        "papers": ["2305.02997"],  # McElfresh
    },
    {
        "q": "How well do simple, well-tuned MLPs perform on tabular benchmarks?",
        "papers": ["2106.11189"],  # Kadra - Well-tuned Simple Nets
    },

    # --- Gradient boosting foundations ---
    {
        "q": "How does XGBoost handle sparsity and scale tree boosting efficiently?",
        "papers": ["1603.02754"],  # Chen & Guestrin
    },
    {
        "q": "How does CatBoost reduce prediction shift when encoding categorical features?",
        "papers": ["1706.09516"],  # Prokhorenkova - ordered boosting / ordered target statistics
    },

    # --- Attention / transformer architectures ---
    {
        "q": "How does the FT-Transformer tokenize numerical and categorical features?",
        "papers": ["2106.11959"],  # Gorishniy - Revisiting DL
    },
    {
        "q": "How does TabNet use sequential attention to select features at each decision step?",
        "papers": ["1908.07442"],  # Arik & Pfister
    },
    {
        "q": "How does SAINT's inter-sample (row) attention differ from feature attention?",
        "papers": ["2106.01342"],  # Somepalli
    },
    {
        "q": "How does TabTransformer use contextual embeddings for categorical features?",
        "papers": ["2012.06678"],  # Huang
    },
    {
        "q": "How does AutoInt learn feature interactions with self-attention?",
        "papers": ["1810.11921"],  # Song
    },
    {
        "q": "How does attention operate between datapoints rather than within a single row?",
        "papers": ["2106.02584"],  # Kossen - NPT
    },
    {
        "q": "How does T2G-Former organize tabular features into relation graphs?",
        "papers": ["2211.16887"],  # Yan
    },

    # --- Boosting-inspired / tree-mimic neural nets ---
    {
        "q": "What makes GrowNet a boosting-inspired neural architecture?",
        "papers": ["2002.07971"],  # Badirli - gradient boosting of shallow nets
    },
    {
        "q": "How do Neural Oblivious Decision Ensembles (NODE) make tree-style splits differentiable?",
        "papers": ["1909.06312"],  # Popov
    },
    {
        "q": "How can a decision-tree structure be embedded as a differentiable neural network layer?",
        "papers": ["2002.07772", "1909.06312"],  # Hazimeh Tree Ensemble Layer, or NODE
    },
    {
        "q": "How can a neural network be distilled into a soft decision tree for interpretability?",
        "papers": ["1711.09784"],  # Frosst & Hinton
    },

    # --- Feature embeddings ---
    {
        "q": "Why do embeddings for numerical features improve tabular deep learning models?",
        "papers": ["2203.05556"],  # Gorishniy - numerical embeddings
    },
    {
        "q": "How does parameter-efficient ensembling of MLPs (TabM) improve tabular performance?",
        "papers": ["2410.24210"],  # Gorishniy - TabM
    },

    # --- Foundation models ---
    {
        "q": "How does TabPFN solve small tabular classification problems in a single forward pass?",
        "papers": ["2207.01848"],  # Hollmann
    },

    # --- Survey-level (broader; may legitimately hit a survey OR a primary paper) ---
    {
        "q": "What are the main categories of deep learning methods proposed for tabular data?",
        "papers": ["2110.01889", "2504.16109"],  # Borisov survey / Representation-learning survey
    },
]
