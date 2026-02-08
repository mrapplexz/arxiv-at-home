# Arxiv at Home

A self-hosted, modular semantic search engine for academic papers.

## Features

* **Advanced Hybrid Search**:
    * **Retrieval**: Combines **Dense Vector Retrieval** (Embeddings) and **Sparse Vector Retrieval** (BM-25) for robust
      candidate generation.
    * **Generative Reranking**: Utilizes Causal LLMs (e.g., Qwen) in a pointwise fashion to score relevance based on
      token probabilities.
    * **Citation Boosting**: Optionally modulates semantic scores using a logarithmic citation
      boost to surface highly impactful papers.
* **Modular Architecture**:
    * **Pluggable Ingestion**: The system is designed to allow ingestion from
      multiple sources. Currently supports only **arXiv Metadata Dump from Kaggle**.
    * **Pluggable Citations**: Supports **Semantic Scholar** for real-time citation counts, with a **NoOp** fallback for
      fully offline/isolated deployments.
* **Robust Data Consistency**:
    * **ACID-Compliant Indexing**: A specific "Reservation" system in PostgreSQL uses row-level locking to ensure
      exactly-once indexing. This allows multiple indexer workers to run
      concurrently without race conditions.
    * **Incremental Sync**: Tracks synchronization state per source, allowing for efficient daily updates without
      re-ingesting the entire dataset.
* **High Availability**: The API is stateless and concurrency-safe; all state is persistent in PostgreSQL and Qdrant.

# Installation (Source Code)

## Prerequisites

* Docker & Docker Compose
* Python 3.11+
* [uv](https://github.com/astral-sh/uv) (for dependency management)
* *Recommended*: NVIDIA GPU with CUDA support for Indexing and Reranking.

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/mrapplexz/arxiv-at-home.git
   cd arxiv-at-home
   ```

2. **Start Infrastructure**

   Launch the required services (Qdrant, PostgreSQL) using Docker Compose:

   ```bash
   docker compose up -d
   ```

3. **Configure Environment**

   Copy the example environment file and edit it:

   ```bash
   cp .env.example .env
   # edit .env
   ```

4. **Install Dependencies**

   ```bash
   uv sync
   ```

## Usage

### 1. Database Migration

Initialize the database schema:

```bash
uv run python -m arxiv_at_home.migrate
```

### 2. Sync Metadata

Import paper metadata into the database.

Currently, the system supports the **Kaggle Arxiv Dataset**.

1. Download `arxiv-metadata-oai-snapshot.json` from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
2. Configure `example/sync.json`.
    * **Note**: You can define `filter_categories` (e.g., `["cs.AI", "cs.LG"]`) to only ingest specific domains.

```bash
uv run python -m arxiv_at_home.sync --config-path example/sync.json
```

You can safely re-run this module to sync with a new Kaggle dump file. The system will upsert only modified paper
metadata.

### 3. Index Papers

Generate embeddings and index them.

The indexer pulls papers from the sync stage that have not been indexed yet. It uses a "Reservation" mechanism, meaning
you can stop and restart the process at any time, or run multiple indexers in parallel.

```bash
uv run python -m arxiv_at_home.index --config-path example/index.json
```

### 4. Run the API

Start the REST API server to serve search traffic.

```bash
uv run python -m arxiv_at_home.api --config-path example/api.json
```

The API will be available at `http://localhost:1337` with default configuration.

Note that you can run multiple API instances safely to scale horizontally - it is stateles.

## Running via Docker Image

If you prefer not to set up a local Python environment, you can run the application components using the pre-built Docker image: `mrapplexz/arxiv-at-home`.

**Prerequisites for Docker Run:**
* Ensure your `.env` file lists the database host as accessible from the container (e.g., `host.docker.internal` for Docker Desktop, or use `--network host` on Linux).
* In the examples below, we mount the current directory (`$(pwd)`) to `/data` inside the container. You must update your config JSON files (e.g., `example/sync.json`) to point to file paths inside `/data/`.

### 1. Database Migration

```bash
docker run --rm --network host --env-file .env \
  mrapplexz/arxiv-at-home arxiv_at_home.migrate
```

### 2. Sync Metadata

Ensure the Kaggle JSON dump is in your current directory (or adjust the volume mount).

```bash
docker run --rm --network host --env-file .env \
  -v "$(pwd):/data" \
  mrapplexz/arxiv-at-home arxiv_at_home.sync --config-path /data/example/sync.json
```

### 3. Index Papers (GPU Recommended)

If you have an NVIDIA GPU, add the `--gpus all` flag:

```bash
docker run --rm --network host --gpus all --env-file .env \
  -v "$(pwd):/data" \
  mrapplexz/arxiv-at-home arxiv_at_home.index --config-path /data/example/index.json
```

### 4. Run the API

```bash
docker run --rm --network host --gpus all --env-file .env \
  -v "$(pwd):/data" \
  -p 1337:1337 \
  mrapplexz/arxiv-at-home arxiv_at_home.api --config-path /data/example/api.json
```


## Configuration Reference

Please see the Pydantic model definitions:

* [Migrations](https://github.com/mrapplexz/arxiv-at-home/blob/main/src/arxiv_at_home/migrate/settings.py)
* [Sync](https://github.com/mrapplexz/arxiv-at-home/blob/main/src/arxiv_at_home/sync/settings.py)
* [Index](https://github.com/mrapplexz/arxiv-at-home/blob/main/src/arxiv_at_home/index/settings.py)
* [API](https://github.com/mrapplexz/arxiv-at-home/blob/main/src/arxiv_at_home/api/settings.py)

## Architecture Details

### Search Workflow

1. **Vectorization**: The user query is tokenized and embedded using the configured Dense Vectorizer.
2. **Qdrant Retrieval**: A fused query is sent to Qdrant:
    * `metadata/dense`
    * `metadata/sparse` (BM25 with IDF - it uses internal `fastembed` implementation)
    * Fused via `Fusion.DBSF` (Distribution-Based Score Fusion).
3. **Hydration**: Full paper metadata is retrieved from the storage database based on the IDs returned by Qdrant.
4. **Citation Context**: Citation counts are fetched from the configured provider (e.g., Semantic Scholar).
5. **Reranking**:
    1. **Semantic**: The Causal LLM scores the `(Query, Paper)` pair.
    2. **Boosting**: The score is adjusted: `Final = SemanticScore * (1 + weight * log10(Citations + 1))`.
6. **Response**: The top `k` results are returned.

## Limitations and Future Work

### Data Ingestion Pipelines

Currently, we implement the synchronization provider that relies on periodic snapshots from Kaggle datasets. While
effective for prototyping, this introduces a synchronization latency.
Our project architecture involves a `PaperMetadataProvider` abstraction. Future iterations may implement an ingestion
pipeline based on the **Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH)**, facilitating direct and
immediate synchronization with arXiv servers.

### Citation Topology

A significant constraint in the current system is the reliance on the Semantic Scholar API for citation metrics. This
design choice was necessitated by the constraints of accessing arXiv's "requester-pays" S3 buckets for bulk source file
retrieval.
In future work, one may implement a full-dump synchronization strategy that ingests raw TeX sources. This will enable
the construction of a proprietary citation graph, allowing for more accurate and cost-effective citation topology
estimation locally.

### Domain-Specific Embeddings

We currently employ generic Qwen models for vector representation and neural reranking. While effective, these models
lack the granular understanding required for niche scientific domains. To address this, we plan to train custom
embedding and reranking models. By utilizing synthetic data generation to simulate complex scientific queries, we
anticipate substantial gains in retrieval performance.

## Development

### Code Quality

The project uses `ruff` for linting and formatting.

```bash
uv run ruff check
uv run ruff format
```
```