from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from d9d.dataset import PaddingSide1D, pad_stack_1d
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

from arxiv_at_home.api.component.reranker.config import RerankerConfig


class RerankInputs(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class RerankInputProcessor:
    def __init__(self, tokenizer: Tokenizer, device: str) -> None:
        self._tokenizer = tokenizer
        self._device = device

    def _encode_template(self, template: str) -> RerankInputs:
        input_tokens = self._tokenizer.encode(template)
        return {
            "input_ids": torch.tensor(
                input_tokens.ids,
                dtype=torch.long,
                device=self._device,
            ),
            "attention_mask": torch.tensor(
                input_tokens.attention_mask,
                dtype=torch.long,
                device=self._device,
            ),
        }

    def encode(self, templates: list[str]) -> RerankInputs:
        encodings = [self._encode_template(x) for x in templates]
        return {
            "input_ids": pad_stack_1d(
                [x["input_ids"] for x in encodings], pad_value=0, padding_side=PaddingSide1D.left
            ),
            "attention_mask": pad_stack_1d(
                [x["attention_mask"] for x in encodings], pad_value=0, padding_side=PaddingSide1D.left
            ),
        }


class GenerativeReranker:
    def __init__(self, config: RerankerConfig, model: AutoModelForCausalLM, tokenizer: Tokenizer) -> None:
        self._config = config
        self._device = torch.device(config.device)
        self._model = model

        # Cache token IDs for scoring
        self._token_true_id = tokenizer.token_to_id(config.token_true)
        self._token_false_id = tokenizer.token_to_id(config.token_false)

    @torch.inference_mode()
    def __call__(self, batch: RerankInputs) -> list[float]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits

        batch_scores = logits[:, -1, :]

        true_vector = batch_scores[:, self._token_true_id]
        false_vector = batch_scores[:, self._token_false_id]

        # Stack as [False, True]
        relevant_logits = torch.stack([false_vector, true_vector], dim=1)

        # Normalize via LogSoftmax
        log_probs = F.log_softmax(relevant_logits, dim=1)

        # Exp(Index 1) gives validation probability (P(True))
        scores = log_probs[:, 1].exp().tolist()

        return scores
