from collections.abc import Generator
from contextlib import contextmanager

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from arxiv_at_home.api.component.reranker.config import RerankerConfig
from arxiv_at_home.api.component.reranker.model import GenerativeReranker, RerankInputProcessor
from arxiv_at_home.api.component.reranker.template import RerankTemplate


def _create_tokenizer(config: RerankerConfig) -> Tokenizer:
    return AutoTokenizer.from_pretrained(config.model).backend_tokenizer


@contextmanager
def create_reranker(config: RerankerConfig) -> Generator[GenerativeReranker, None, None]:
    model = (
        AutoModelForCausalLM.from_pretrained(config.model, dtype=torch.bfloat16, attn_implementation="sdpa")
        .eval()
        .to(config.device)
    )

    yield GenerativeReranker(config, model, _create_tokenizer(config))


def create_rerank_processor(config: RerankerConfig) -> RerankInputProcessor:
    return RerankInputProcessor(_create_tokenizer(config), config.device)


def create_rerank_template(config: RerankerConfig) -> RerankTemplate:
    return RerankTemplate(config)
