# Scikit-learn Deep Notes — Pipelines, RAG, MCP, LLM Processing & Regression

---

## 1. Pipelines — Minute-Level Detail

### 1.1 What a Pipeline actually does internally
A `Pipeline` is a list of `(name, transformer/estimator)` steps. Internally:
- On `.fit(X, y)`: calls `.fit_transform()` on every step **except the last**, passing the output forward as input to the next step. The last step gets `.fit()` (or `.fit_transform()` if it's also a transformer).
- On `.predict(X)`: calls `.transform()` on every step except the last, then `.predict()` on the last.
- This means **every intermediate step must implement `fit`/`transform`**, and only the final step needs `predict` (if it's a predictive pipeline).

```python
Pipeline([
    ("imputer", SimpleImputer()),      # fit_transform
    ("scaler", StandardScaler()),      # fit_transform
    ("model", LogisticRegression())    # fit only (final step)
])
```

### 1.2 Why Pipelines exist — the leakage problem in detail
Without a pipeline, a common mistake:
```python
scaler.fit(X)                # fit on ENTIRE dataset — sees test set statistics
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled, ...)
```
This leaks test-set mean/variance into training. With cross-validation this gets worse — each fold needs its **own** independent fit. A `Pipeline` passed directly into `cross_val_score` or `GridSearchCV` automatically refits preprocessing on each fold's training portion only. This is the #1 reason pipelines are considered mandatory in production, not optional convenience.

### 1.3 `ColumnTransformer` — parallel branches
Unlike `Pipeline` (sequential), `ColumnTransformer` applies **different transformers to different columns in parallel**, then concatenates outputs:
```python
ColumnTransformer([
    ("num", numeric_pipeline, ["age", "income"]),
    ("txt", TfidfVectorizer(), "description"),   # note: single string column, not list
    ("cat", OneHotEncoder(), ["city"]),
])
```
This is the piece that lets you mix tabular + text + categorical features into one model — directly relevant if you ever build a hybrid system that blends structured metadata with embedded text.

### 1.4 `FeatureUnion` vs `ColumnTransformer`
- `FeatureUnion`: combines multiple transformers applied to the **same full input**, concatenates their outputs (e.g., TF-IDF features + hand-crafted features from the same text).
- `ColumnTransformer`: routes **different columns** to different transformers. Almost always what you want for tabular + mixed-type data.

### 1.5 Custom pipeline steps
Any class with `fit`/`transform` works — this is how you'd insert a custom embedding step (e.g., calling an embedding model) directly into a scikit-learn pipeline:
```python
from sklearn.base import BaseEstimator, TransformerMixin

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return [self.embed_fn(text) for text in X]
```
`TransformerMixin` auto-generates `fit_transform` from `fit` + `transform`. `BaseEstimator` gives you `get_params`/`set_params` for free — required if you want `GridSearchCV` to tune this step.

### 1.6 Caching expensive steps
```python
from tempfile import mkdtemp
Pipeline([...], memory=mkdtemp())
```
Caches transformer outputs to disk — critical if step 1 (e.g., embedding generation) is expensive and you're grid-searching over the downstream model only. Without this, every grid search iteration re-embeds everything.

---

## 2. RAG (Retrieval-Augmented Generation) — Where Scikit-learn Fits

RAG itself (retrieval + generation) isn't built with scikit-learn — that's ChromaDB/vector DB + LLM territory. But scikit-learn does real work **around** the RAG pipeline:

### 2.1 Similarity computation
```python
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

sim_matrix = cosine_similarity(query_embedding, document_embeddings)
```
- Vector DBs (ChromaDB) do this internally at scale with approximate nearest neighbor (ANN) indexing, but for debugging small batches, prototyping a retrieval scorer, or reranking a shortlist manually, this is the direct tool.
- Useful when you want to **rerank** ChromaDB's top-k results with a different similarity metric than the DB used at index time.

### 2.2 Dimensionality reduction for embedding visualization
```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

reduced = PCA(n_components=2).fit_transform(embeddings)   # fast, linear
reduced = TSNE(n_components=2, perplexity=30).fit_transform(embeddings)  # slower, preserves local structure
```
- **Debugging retrieval quality**: plot your document chunk embeddings in 2D. If chunks from the same source document/topic don't cluster, your chunking strategy or embedding model may be misconfigured.
- **PCA** is linear and fast — good first pass, preserves global variance structure.
- **t-SNE** is nonlinear, better at revealing local clusters, but slower and coordinates aren't globally meaningful (don't compare distances across a t-SNE plot the way you would raw embeddings).

### 2.3 Clustering retrieved/indexed documents
```python
from sklearn.cluster import KMeans, DBSCAN

labels = KMeans(n_clusters=10).fit_predict(document_embeddings)
```
- **Topic discovery**: cluster your entire ChromaDB corpus's embeddings to see what topical groups exist — useful for auditing what your knowledge base actually covers vs. what you think it covers.
- **Deduplication**: `DBSCAN` (density-based, doesn't need a predefined cluster count) can flag near-duplicate chunks sitting extremely close in embedding space — common issue when documents get chunked with overlapping windows.
- **Diverse retrieval (MMR-style)**: after retrieving top-k by similarity, cluster them and sample one from each cluster to reduce redundancy in what gets passed to the LLM context.

### 2.4 Evaluating retrieval quality
```python
from sklearn.metrics import ndcg_score, precision_score, recall_score
```
- If you have labeled "correct document for this query" data, precision@k / recall@k / NDCG (ranking quality) from scikit-learn's metrics module give you a quantifiable retrieval score — this is how you'd A/B test two chunking strategies or two embedding models objectively instead of eyeballing results.

### 2.5 Classical retrieval as a fallback/hybrid signal
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- TF-IDF + cosine similarity is the classic **sparse/keyword-based retrieval** method (BM25's cousin). Production RAG systems often run **hybrid search**: dense embedding similarity + TF-IDF/BM25 keyword score, combined (e.g., weighted sum or reciprocal rank fusion). Scikit-learn's `TfidfVectorizer` is the standard way to generate the sparse side of that hybrid signal.
- Why this matters: dense embeddings are bad at exact keyword/entity matches (e.g., product codes, exact names); TF-IDF catches what embeddings miss.

### 2.6 Routing queries (classical classifier in front of RAG)
```python
from sklearn.linear_model import LogisticRegression
```
- Train a lightweight classifier on query text (TF-IDF features) to decide: does this query need retrieval at all, or can it be answered directly? Does it need the SQL tool vs. the vector store?
- This is cheaper and faster than an LLM-based router call, and is a legitimate production pattern for cutting latency/cost in agent systems.

---

## 3. MCP (Model Context Protocol) — Where Scikit-learn Fits

MCP itself is a protocol for connecting LLM agents to tools/data sources — it's not something scikit-learn plugs into directly. But in systems built **around** MCP servers, classical ML shows up in supporting roles:

### 3.1 Tool selection / routing classifiers
When an agent has many MCP tools available (file access, DB query, web search, etc.), and latency/cost matters, a lightweight `RandomForestClassifier` or `LogisticRegression` trained on past query→tool-used pairs can pre-filter or rank which MCP tools are likely relevant before the LLM even reasons about it — reducing the number of tool descriptions stuffed into context.

### 3.2 Anomaly detection on tool call patterns
```python
from sklearn.ensemble import IsolationForest
```
- If you're logging MCP tool call sequences (which tools get called, with what parameters, how often), `IsolationForest` or `OneClassSVM` can flag anomalous agent behavior — e.g., a tool being called with malformed parameters repeatedly, or an unusual sequence that might indicate the agent is stuck in a loop or misbehaving. This is a genuine production monitoring pattern for agentic systems.

### 3.3 Evaluating MCP server response quality
- If an MCP tool returns structured data (e.g., search results, file contents) and you want to score whether the agent's downstream usage of that data was "correct," scikit-learn's classification metrics (precision/recall/F1) apply the same way as any other evaluation task — you're just evaluating agent-tool-output pairs instead of a plain ML model.

### 3.4 Feature engineering from MCP-sourced data
- If an MCP server retrieves structured data (e.g., a database row, a file's metadata) that then feeds into a classical model downstream (not the LLM), all the standard scikit-learn preprocessing tools (`ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`) apply exactly as they would with any tabular data source — MCP is just the data-fetching layer in front of it.

**Honest note:** MCP is a fairly recent protocol focused on LLM-tool connectivity, and scikit-learn has no native or special integration with it — its role here is always as a supporting classical-ML layer around the agent system, not something that touches MCP directly.

---

## 4. LLM Processing — Where Scikit-learn Fits

### 4.1 Pre-processing text before it reaches the LLM
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
```
- **Deduplication of input documents** before chunking/embedding (cosine similarity on TF-IDF vectors is a cheap first-pass dedup filter before you pay for embedding generation).
- **Length/complexity triage**: simple feature extraction (word count, TF-IDF sparsity, readability scores) to route documents to different chunking strategies.

### 4.2 Classical NLP baselines vs. LLM calls
- Before using an LLM for a classification task (sentiment, intent, topic labeling), build `TfidfVectorizer` + `LogisticRegression`/`MultinomialNB` baseline:
  - If baseline hits 90%+ accuracy, an LLM call is probably unjustified cost/latency for that task.
  - If baseline caps at 60%, that's your evidence the task genuinely needs LLM-level reasoning (context, nuance, world knowledge).
- This comparison is a standard, expected step in real ML engineering — never skip straight to "throw an LLM at it."

### 4.3 Post-processing LLM outputs
```python
from sklearn.metrics import classification_report
```
- If your LLM produces structured/classification-style outputs (e.g., labeling support tickets, extracting categories), you still evaluate it with the exact same classification metrics as a classical model — precision, recall, F1, confusion matrix — against a human-labeled test set. LLM-based systems don't get a pass on rigorous evaluation.

### 4.4 Guardrails / output classifiers
- A small, fast scikit-learn classifier can act as a **guardrail layer** in front of or behind an LLM: e.g., a toxicity/PII classifier trained on TF-IDF features that runs in milliseconds, versus calling another LLM for moderation (slower, costlier). Many production systems use this exact "cheap classifier as guardrail, LLM as the reasoning engine" split.

### 4.5 Confidence/uncertainty modeling
```python
model.predict_proba(X)
```
- When you have a classical component alongside your LLM (e.g., an intent classifier before routing to an LLM prompt), `predict_proba` gives calibrated confidence scores you can threshold on — e.g., "if confidence < 0.6, fall back to asking the LLM to disambiguate."

### 4.6 Feature extraction from embeddings for lightweight downstream tasks
- Embeddings from your LLM/embedding model are just numeric vectors once generated — scikit-learn is completely agnostic to where they came from. Any classifier/regressor/clusterer can be trained **on top of** embeddings as input features (e.g., `LogisticRegression` trained on sentence embeddings for a classification task — often outperforms TF-IDF and is much cheaper than fine-tuning or repeated LLM calls at inference time).

---

## 5. Regression — Full Depth

### 5.1 Linear Regression — the foundation
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
- Minimizes **sum of squared residuals** (Ordinary Least Squares).
- `model.coef_` = learned weights, `model.intercept_` = bias term.
- Assumes a **linear relationship** between features and target, **homoscedasticity** (constant error variance), and **no severe multicollinearity** between features.

### 5.2 Regularized regression — solving overfitting
| Model | Penalty | Effect |
|---|---|---|
| `Ridge` | L2: adds `alpha * sum(coef^2)` to loss | Shrinks all coefficients smoothly toward zero, keeps all features, handles multicollinearity well |
| `Lasso` | L1: adds `alpha * sum(abs(coef))` to loss | Can zero out coefficients entirely → automatic feature selection |
| `ElasticNet` | Mix of L1 + L2 (`l1_ratio` param) | Balances feature selection with coefficient stability; good when features are correlated (Lasso alone can behave erratically with correlated features) |

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
Ridge(alpha=1.0)         # higher alpha = more regularization = simpler model
Lasso(alpha=0.1)
ElasticNet(alpha=0.1, l1_ratio=0.5)   # l1_ratio=1 → pure Lasso, l1_ratio=0 → pure Ridge
```
**Tuning `alpha`**: use `RidgeCV`/`LassoCV` which cross-validate over a range of alpha values automatically.

### 5.3 Nonlinear regression
| Model | How it captures nonlinearity |
|---|---|
| `PolynomialFeatures` + `LinearRegression` | Explicitly expands features into polynomial terms, then fits linearly on the expanded space |
| `DecisionTreeRegressor` | Splits feature space into regions, predicts the mean per region |
| `RandomForestRegressor` | Averages many decision trees, reduces overfitting vs. single tree |
| `GradientBoostingRegressor` | Sequentially fits trees to residual errors — often best raw performance on tabular data |
| `SVR` (Support Vector Regression) | Fits within an epsilon-margin tube, kernel trick allows nonlinear boundaries |

### 5.4 Evaluation metrics — what each one actually tells you
| Metric | Formula intuition | Interpretation caveat |
|---|---|---|
| MAE | Mean absolute error | Same units as target, robust to outliers |
| MSE | Mean squared error | Penalizes large errors heavily (squared), sensitive to outliers |
| RMSE | sqrt(MSE) | Same units as target, still outlier-sensitive |
| R² | 1 − (residual variance / total variance) | % of variance explained; **can be negative** if model is worse than predicting the mean |
| MAPE | Mean absolute percentage error | Scale-independent, but breaks down near y=0 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

### 5.5 Diagnosing a regression model (beyond just the metric number)
- **Residual plots**: plot `y_test - y_pred` against `y_pred`. Random scatter around zero = good fit. A visible pattern (curve, funnel shape) = model is missing structure (nonlinearity) or violating homoscedasticity.
- **Multicollinearity check**: high correlation between input features inflates coefficient variance in linear models — check with a correlation matrix or Variance Inflation Factor (VIF, from `statsmodels`, not sklearn) before trusting `LinearRegression` coefficients for interpretation.

### 5.6 Where regression shows up in AI-engineering-adjacent work
- Predicting latency/cost of an LLM call given prompt length + model choice (a regression problem you could genuinely build).
- Predicting embedding similarity scores as a continuous target when building a custom reranker.
- Estimating confidence/quality scores for generated outputs as a continuous target (rather than classification), if you have graded human feedback data.

---

## 6. Cross-Cutting Reference — Module Map

| Module | What lives there |
|---|---|
| `sklearn.preprocessing` | Scalers, encoders, imputers |
| `sklearn.pipeline` | `Pipeline`, `FeatureUnion` |
| `sklearn.compose` | `ColumnTransformer` |
| `sklearn.linear_model` | Linear/Logistic/Ridge/Lasso |
| `sklearn.ensemble` | RandomForest, GradientBoosting, IsolationForest, AdaBoost |
| `sklearn.svm` | SVC, SVR |
| `sklearn.tree` | DecisionTree |
| `sklearn.neighbors` | KNN |
| `sklearn.naive_bayes` | GaussianNB, MultinomialNB |
| `sklearn.cluster` | KMeans, DBSCAN, Agglomerative |
| `sklearn.decomposition` | PCA, TruncatedSVD, NMF |
| `sklearn.manifold` | t-SNE |
| `sklearn.feature_extraction.text` | TfidfVectorizer, CountVectorizer |
| `sklearn.metrics` | All evaluation metrics, `pairwise` similarity functions |
| `sklearn.model_selection` | train_test_split, cross_val_score, GridSearchCV |
| `sklearn.base` | `BaseEstimator`, `TransformerMixin` for custom components |

---

## 7. One-Paragraph Summary (for quick recall)

Scikit-learn is the classical-ML backbone that sits **around** your LLM/RAG/MCP systems rather than inside them: it powers pipeline discipline (preventing data leakage via `Pipeline`/`ColumnTransformer`), gives you the tools to debug and evaluate retrieval quality in RAG (similarity, clustering, dimensionality reduction on embeddings, TF-IDF hybrid search), supplies cheap classifiers that act as routers/guardrails in front of expensive LLM calls (including MCP tool-selection and agent-monitoring use cases), and remains the standard toolkit for regression and evaluation metrics whenever you need fast, interpretable, CPU-only models — either as a baseline to justify LLM cost, or as a genuine production component in a cost-optimized hybrid system.
