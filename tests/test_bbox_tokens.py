from pathlib import Path

import pytest

from olmocr.prompts.prompts import PageResponse
from olmocr.train.dataloader import FrontMatterParser, normalize_bbox_markers
from olmocr.train.train import ensure_bbox_special_tokens


class DummyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0}
        self.added_special_tokens: list[str] = []

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def add_special_tokens(self, tokens: dict[str, list[str]]) -> int:
        additional = tokens.get("additional_special_tokens", [])
        for tok in additional:
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
                self.added_special_tokens.append(tok)
        return len(additional)

    def encode(self, text: str) -> list[int]:
        return [self._vocab[text]] if text in self._vocab else []


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()


@pytest.mark.parametrize("token_key", ["[BBOX_BLK_END]"])
def test_ensure_bbox_special_tokens_adds_missing_tokens(token_key):
    processor = DummyProcessor()

    added = ensure_bbox_special_tokens(processor)
    assert added is True
    vocab = processor.tokenizer.get_vocab()
    assert token_key in vocab

    # Calling again should be a no-op since tokens already exist
    assert ensure_bbox_special_tokens(processor) is False


def test_normalize_bbox_markers_transforms_sample_text():
    sample_dir = Path("tests/sample_dataset_bbox/28921348-28921360-page-12")
    md_path = sample_dir / "28921348-28921360-page-12.md"
    parser = FrontMatterParser(front_matter_class=PageResponse)

    sample = {"markdown_path": md_path}
    processed = parser(sample)
    assert processed is not None
    natural_text = processed["page_data"].natural_text

    assert natural_text is not None
    assert "<!-- bbox" in natural_text

    normalized_text = normalize_bbox_markers(natural_text)
    assert normalized_text is not None
    assert "<|box_start|>" in normalized_text
    assert "<|box_end|>" in normalized_text


def test_bbox_tokens_encode_after_registration():
    processor = DummyProcessor()

    assert ensure_bbox_special_tokens(processor)

    tokenizer = processor.tokenizer
    blk_id = tokenizer.encode("[BBOX_BLK_END]")

    assert blk_id == [tokenizer.get_vocab()["[BBOX_BLK_END]"]]
