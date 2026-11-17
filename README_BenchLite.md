# BenchLite --- A Lightweight Python Toolkit for RAG Evaluation

BenchLite is a Python package for evaluating selected retrieval and
answer-quality aspects of Retrieval-Augmented Generation (RAG) systems.
It provides a centralized evaluator and modular metric implementations
for retrieval ranking, semantic answer similarity, and a simple lexical
context-grounding heuristic.

The package is distributed on PyPI as `benchlite`. The Python import
package in the current repository is named `evalrag`.

> **Scope:** BenchLite is an evaluation utility, not a complete RAG
> application or vector database. It evaluates inputs supplied by the
> user; it does not itself build a document-ingestion, embedding, or
> retrieval pipeline.

## Current capabilities

The current implementation includes:

  -----------------------------------------------------------------------
  Metric                              What it measures
  ----------------------------------- -----------------------------------
  Recall@K                            Fraction of annotated relevant
                                      documents retrieved in the top K
                                      results

  Precision@K                         Fraction of returned top-K
                                      documents that are annotated
                                      relevant

  Mean Reciprocal Rank (MRR@K)        Reciprocal rank of the first
                                      relevant result, averaged across
                                      queries

  Answer similarity                   Mean cosine similarity between
                                      sentence embeddings of predictions
                                      and reference answers

  Lexical grounding heuristic         Fraction of prediction
                                      unigram/bigram features absent from
                                      the paired context
  -----------------------------------------------------------------------

### How it works

1.  Provide a list of queries, retrieved results, and reference answers.
2.  Optionally provide relevance annotations for retrieval metrics.
3.  Optionally provide a prediction function that generates an answer
    from a query and its retrieved text.
4.  The evaluator computes the applicable metrics.
5.  Results are returned as a one-row Pandas DataFrame containing
    aggregate values.

## Architecture

``` text
Queries + retrieved documents + reference answers
                     |
                     v
             Input normalization
                     |
          +----------+-----------+
          |                      |
          v                      v
   Retrieval metrics       Optional prediction function
   Recall@K                 (query, retrieved texts)
   Precision@K                       |
   MRR@K                              v
                              Generated predictions
                                      |
                             +--------+--------+
                             |                 |
                             v                 v
                     Answer similarity   Context-grounding
                     SentenceTransformers heuristic
                             |                 |
                             +--------+--------+
                                      |
                                      v
                             Aggregate results
                                Pandas DataFrame
```

## Installation

Install the published distribution:

``` bash
pip install benchlite
```

The source package currently uses the `evalrag` import namespace. For
example:

``` python
from evalrag.evaluator import Evaluator
```

If installation or import behaviour differs for the published release,
consult the PyPI release metadata and the repository version before use.

## Quick start

The following example illustrates the evaluator interface implemented in
the repository. Replace the sample values with the document IDs,
relevance annotations, and reference answers from your own RAG
evaluation dataset.

``` python
from evalrag.evaluator import Evaluator

queries = [
    "What is retrieval-augmented generation?",
    "What does a vector database store?"
]

# Each inner list contains ranked document identifiers or text.
retrieveds = [
    ["doc_1", "doc_3", "doc_2"],
    ["doc_4", "doc_5", "doc_6"]
]

references = [
    "Retrieval-augmented generation combines retrieval with text generation.",
    "A vector database stores vector embeddings and supports similarity search."
]

# Each set contains the relevant document IDs for the corresponding query.
id_to_relevant = [
    {"doc_1", "doc_2"},
    {"doc_4"}
]

evaluator = Evaluator(k=3)

results = evaluator.evaluate(
    queries=queries,
    retrieveds=retrieveds,
    references=references,
    id_to_relevant=id_to_relevant
)

print(results)
```

This example computes retrieval metrics. Answer similarity and the
context-grounding heuristic require predictions, which can be supplied
through `prediction_fn`; the grounding heuristic also requires
`contexts`.

### Evaluating generated answers

The `prediction_fn` argument is a callable that receives a query and its
retrieved text list and returns a generated answer string.

``` python
def my_prediction_function(query, retrieved_texts):
    # Connect your own RAG generation step here.
    # Return the generated answer as a string.
    return "Your generated answer"

contexts = [
    "Context used to generate the first answer.",
    "Context used to generate the second answer."
]

results = evaluator.evaluate(
    queries=queries,
    retrieveds=retrieveds,
    references=references,
    id_to_relevant=id_to_relevant,
    contexts=contexts,
    prediction_fn=my_prediction_function
)

print(results)
```

The example function above is a placeholder. Connect it to your own
language model or RAG generation pipeline to evaluate real generated
answers.

## Input format

-   `queries`: list of query strings.
-   `retrieveds`: list of ranked result lists. Each result can be a
    document ID/text or a tuple/list whose first element is the document
    ID/text.
-   `references`: list of reference answer strings.
-   `id_to_relevant`: optional list of sets containing relevant document
    IDs for each query.
-   `prediction_fn`: optional callable with signature
    `(query, retrieved_texts) -> predicted_answer`.
-   `contexts`: optional list of context strings paired with generated
    predictions.
-   `k`: number of top retrieved results considered.

The lengths of `queries`, `retrieveds`, and `references` must match.
Relevance annotations and contexts should also be aligned with the
corresponding queries.

## Metric details and limitations

### Retrieval metrics

Recall@K, Precision@K, and MRR@K use the supplied relevance annotations.
Their usefulness depends on the quality and consistency of those
annotations.

Precision@K in the current implementation divides by the number of
results actually returned when fewer than K results are available. This
convention should be considered when comparing results with other
evaluation tools.

### Answer similarity

Answer similarity uses Sentence Transformers (default model:
`all-MiniLM-L6-v2`) to embed predictions and reference answers, then
calculates cosine similarity. The implementation maps cosine values from
\[-1, 1\] to \[0, 1\] for reporting.

Semantic similarity is not the same as factual correctness. A high
similarity score does not prove that an answer is correct or fully
supported by the source context.

### Context-grounding heuristic

The current grounding metric compares unigram and bigram features in
each prediction with those in its paired context. It reports the
proportion of prediction n-gram features that are absent from the
context.

This is a lightweight lexical heuristic, not a complete hallucination
detector or a semantic faithfulness metric. It can flag paraphrases as
unsupported and can fail to identify incorrect claims that reuse context
vocabulary. Interpret it as an exploratory signal rather than a
definitive factuality score.

## Technology stack

-   **Language:** Python
-   **Data handling:** Pandas, NumPy
-   **Machine learning utilities:** Scikit-learn
-   **Semantic embeddings:** Sentence Transformers
-   **Progress display:** tqdm
-   **Distribution:** PyPI

## Repository structure

``` text
benchlite/
├── README.md
├── pyproject.toml
└── evalrag/
    ├── __init__.py
    ├── evaluator.py
    ├── utils.py
    └── metrics/
        ├── __init__.py
        ├── recall.py
        ├── precision.py
        ├── mrr.py
        ├── similarity.py
        └── hallucination.py
```

## Project status and roadmap

The core Python evaluation library has been implemented and packaged for
distribution. A separate interactive demonstration application is
planned and is currently under development.

Potential future work includes:

-   Developing an interactive demonstration interface for running
    evaluations and inspecting results.
-   Adding per-query metric outputs alongside aggregate scores.
-   Improving validation and error handling for malformed or misaligned
    inputs.
-   Adding automated unit and integration tests.
-   Exploring semantic faithfulness evaluation methods, such as
    natural-language-inference-based approaches or carefully configured
    LLM-as-a-judge evaluation.
-   Running reproducible benchmark experiments and documenting
    comparisons with established RAG evaluation approaches.
-   Expanding documentation with datasets, examples, and API reference
    material.

## Links

-   **GitHub repository:** https://github.com/adityayadav13/benchlite
-   **PyPI package:** https://pypi.org/project/benchlite/

## Author

**Aditya Yadav**

BenchLite is an ongoing project exploring practical evaluation methods
for retrieval-augmented language generation systems.
