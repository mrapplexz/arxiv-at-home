from collections.abc import Generator
from contextlib import contextmanager

from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer

from arxiv_at_home.common.dense.config import DenseVectorizationConfig
from arxiv_at_home.common.dense.template import DenseEncodingTemplate
from arxiv_at_home.common.dense.vectorizer import DenseVectorizer


@contextmanager
def create_dense_vectorizer(config: DenseVectorizationConfig) -> Generator[DenseVectorizer, None, None]:
    model = AutoModel.from_pretrained(config.model, attn_implementation="sdpa").eval().to(config.device)
    yield DenseVectorizer(config, model)


def create_dense_tokenizer(config: DenseVectorizationConfig) -> Tokenizer:
    return AutoTokenizer.from_pretrained(config.model).backend_tokenizer


def create_dense_template(config: DenseVectorizationConfig) -> DenseEncodingTemplate:
    return DenseEncodingTemplate(config)
