import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

PHONEME_LIST = [
    "pau",
    "A",
    "E",
    "I",
    "N",
    "O",
    "U",
    "a",
    "b",
    "by",
    "ch",
    "cl",
    "d",
    "dy",
    "e",
    "f",
    "g",
    "gw",
    "gy",
    "h",
    "hy",
    "i",
    "j",
    "k",
    "kw",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "p",
    "py",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "v",
    "w",
    "y",
    "z",
]
UNVOICED_PHONEME_LIST = ["A", "I", "U", "E", "O", "cl", "pau"]
MORA_PHONEME_LIST = ["a", "i", "u", "e", "o", "N", *UNVOICED_PHONEME_LIST]

ACCENT_SYMBOL_LIST = ["[", "]", "#", "?", "_"]


class Label:
    def __init__(
        self,
        idx: int,
        start_times: list[float],
        end_times: list[float],
        contexts: list[str],
    ) -> None:
        self.idx = idx
        self.start_times = start_times
        self.end_times = end_times
        self.contexts = contexts

        self.is_full_context = False
        if "/" in self.contexts[0]:
            self.is_full_context = True

    def __len__(self) -> int:
        return len(self.contexts)

    def __str__(self) -> str:
        return f"<Label idx={self.idx} len={len(self)}>"

    @staticmethod
    def load_from_path(lab_path: Path, sec_unit: float = 1.0) -> "Label":
        start_times = []
        end_times = []
        contexts = []
        idx = int(re.search(r"\d+", lab_path.stem).group())

        with Path.open(lab_path, "r") as f:
            for line in f:
                start, end, context = line.strip().split()
                contexts.append(context)
                start_times.append(float(start) / sec_unit)
                end_times.append(float(end) / sec_unit)

        return Label(idx, start_times, end_times, contexts)

    def get_alignments(self, sampling_rate: int, hop_length: int) -> tuple[list[int], list[int], float, float]:
        phonemes = []
        durations = []
        start_time, end_time = 0.0, 0.0
        end_index = 0

        for p, s, e in zip(self.get_phonemes(), self.start_times, self.end_times, strict=True):
            if phonemes == []:
                if p != "sil":
                    start_time = s
                else:
                    continue

            phonemes.append(p)

            if p != "sil":
                end_time = e
                end_index = len(phonemes)

            durations.append(
                int(
                    np.round(e * sampling_rate / hop_length) - np.round(s * sampling_rate / hop_length),
                ),
            )

        if start_time >= end_time:
            msg = f"start_time ({start_time}) is larger than end_time ({end_time})."
            raise ValueError(msg)

        phonemes = phonemes[:end_index]
        durations = durations[:end_index]

        return phonemes, durations, start_time, end_time

    def get_text(self) -> str:
        return " ".join(self.get_phonemes())

    def get_phonemes(self) -> Iterator[str]:
        for context in self.contexts:
            if self.is_full_context:
                p3 = re.search(r"\-(.*?)\+", context).group(1)
                if p3 not in PHONEME_LIST and p3 != "sil":
                    msg = f"p3 ({p3}) is not in PHONEME_LIST."
                    raise ValueError(msg)
                yield p3
            else:
                yield context

    def _get_fullcontext_accents(self, i: int, context: str) -> str | None:
        p3 = re.search(r"\-(.*?)\+", context).group(1)
        if p3 not in PHONEME_LIST and p3 != "sil":
            msg = f"p3 ({p3}) is not in PHONEME_LIST."
            raise ValueError(msg)

        if p3 == "sil":
            if i not in (0, len(self.contexts) - 1):
                msg = "sil is not in the first or last position."
                raise ValueError(msg)
            return None
        if p3 == "pau":
            return "_"

        if p3 not in MORA_PHONEME_LIST:
            return "_"

        a1 = re.search(r"/A:([0-9\-]+)\+", context).group(1)
        a2 = re.search(r"\+(\d+)\+", context).group(1)
        a3 = re.search(r"\+(\d+)/", context).group(1)
        f1 = re.search(r"/F:(\d+)_", context).group(1)

        _a2_next_group = re.search(r"\+(\d+)\+", self.contexts[i + 1])
        a2_next = -50
        if _a2_next_group is not None:
            a2_next = _a2_next_group.group(1)

        if (a3 == 1 and a2_next == 1) or i == len(self.contexts) - 2:
            f3 = re.search(r"#(\d+)_", context).group(1)
            if f3 == 1:
                return "?"
            return "#"
        if a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            return "]"
        if a2 == 1 and a2_next == 2:
            return "["

        return "_"

    def get_accents(self) -> Iterator[str]:
        for i, context in enumerate(self.contexts):
            if self.is_full_context:
                accent = self._get_fullcontext_accents(i, context)
                if accent is not None:
                    yield accent
            else:
                yield "_"

    def get_accent_ids(self) -> Iterator[int]:
        for accent in self.get_accents():
            yield ACCENT_SYMBOL_LIST.index(accent)
