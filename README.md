# Arxiv at Home

A self-hosted semantic search engine for Arxiv papers.

## Features

* **Hybrid Search**: Combines semantic retrieval, keyword search with both neural and citation re-ranking for high
  precision.
* **Self-Hosted**: Run everything locally on your own infrastructure (except for optional **Citation Reranking**
  feature).
* **Robust Scalability & Data Integrity**: Designed to ingest metadata from multiple paper sources. The system supports
  efficient incremental index updates, and we ensure data consistency during write operations.
* **High Availability & Redundancy**: The API architecture is stateless and concurrency-safe, allowing it to be deployed
  across multiple instances.

## Prerequisites

* Docker & Docker Compose
* Python 3.11+
* [uv](https://github.com/astral-sh/uv)

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

   Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` if you need to customize database credentials or ports.

4. **Install Dependencies**

   Using `uv`:
   ```bash
   uv sync
   ```

## Usage

The project consists of several components that need to be run in sequence to set up the system.

### 1. Database Migration

Initialize the database schema:

```bash
uv run python -m arxiv_at_home.migrate
```

### 2. Sync Metadata

Import paper metadata into the database. You'll need the Arxiv metadata dataset (e.g.,
`arxiv-metadata-oai-snapshot.json` from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)).

You can re-run this package to do an incremental update - it will work out of the box.

Configure the sync process in `example/sync.json`, ensuring the `path` points to your dataset file.

```bash
uv run python -m arxiv_at_home.sync --config-path example/sync.json
```

### 3. Index Papers

Generate vector embeddings for the papers and index them.

This step requires a modern GPU for reasonable performance, although it can run on CPU (slowly).

You can re-run this package to do an incremental update - it will work out of the box.

Configure the indexing process in `example/index.json`.

```bash
uv run python -m arxiv_at_home.index --config-path example/index.json
```

### 4. Run the API

Start the REST API server.

Configure the API settings in `example/api.json`.

```bash
uv run python -m arxiv_at_home.api --config-path example/api.json
```

The API will be available at `http://localhost:1337` (or whatever port you configured). You can access the interactive
API documentation at `http://localhost:1337/docs`.

## Limitations and Future Work

### Data Ingestion Pipelines

Currently, we implement the only synchronization provider that relies on periodic snapshots from Kaggle datasets.
While effective for prototyping, this introduces a synchronization latency.
Our project architecture supports different paper metadata providers. Future iterations may implement an ingestion
pipeline based on the Open Archives Initiative Protocol for Metadata Harvesting (OAI-PMH), facilitating direct and
immediate synchronization with arXiv servers.

### Citation Topology

A significant constraint in the current system is the reliance on the Semantic Scholar API for citation metrics. This
design choice was necessitated by the financial constraints of accessing arXiv's "requester-pays" S3 buckets for bulk
source file retrieval. In future work, one may implement a full-dump synchronization strategy that ingests raw TeX
sources.
This will enable the construction of a proprietary citation graph, allowing for more accurate and cost-effective
citation topology estimation.

### Domain-Specific Embeddings

We currently employ Qwen3-Embedding-0.6B for vector representation and Qwen3-Reranker-0.6B for neural reranking. While
effective, these models lack the granular understanding required for niche scientific domains. To address this, we plan
to train custom embedding and reranking models. By utilizing synthetic data generation to simulate
complex scientific queries, we anticipate substantial gains in retrieval performance and semantic alignment.

## Development

### Code Quality

The project uses `ruff` for linting and formatting.

```bash
uv run ruff check
uv run ruff format
```
