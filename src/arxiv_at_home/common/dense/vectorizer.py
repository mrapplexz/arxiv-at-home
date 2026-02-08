from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel
from tokenizers import Tokenizer
from transformers import AutoModel, AutoTokenizer


class VectorizerInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class PoolingMode(StrEnum):
    last_token = "last_token"  # noqa: S105
    first_token = "first_token"  # noqa: S105


class DenseVectorizationConfig(BaseModel):
    device: str
    model: str
    pooling: PoolingMode


def pool_tokens(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, mode: PoolingMode) -> torch.Tensor:
    # assume right padding
    match mode:
        case PoolingMode.last_token:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        case PoolingMode.first_token:
            return last_hidden_states[:, -1]
        case _:
            raise ValueError(f"Unknown pooling mode: {mode}")


class DenseVectorizer:
    def __init__(self, config: DenseVectorizationConfig, model: AutoModel) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._model = model

    @torch.inference_mode()
    def __call__(self, batch: VectorizerInputs) -> list[torch.Tensor]:
        input_ids = batch["input_ids"].to(self._device)
        attention_mask = batch["attention_mask"].to(self._device)

        last_hidden_state = self._model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        pooled = pool_tokens(last_hidden_state, attention_mask, self._config.pooling)

        embeddings = F.normalize(pooled, p=2, dim=1)

        return list(embeddings.cpu())


@contextmanager
def create_dense_vectorizer(config: DenseVectorizationConfig) -> Generator[DenseVectorizer, None, None]:
    model = AutoModel.from_pretrained(config.model, attn_implementation="sdpa").eval().to(config.device)
    yield DenseVectorizer(config, model)


def create_dense_tokenizer(config: DenseVectorizationConfig) -> Tokenizer:
    return AutoTokenizer.from_pretrained(config.model).backend_tokenizer
