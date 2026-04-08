from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


ETHNICITY_TO_DOMAIN = {
    "EA": "CHINESE",
    "SEA": "MALAY",
    "SA": "HINDI",
}

ENGLISH_LANGUAGES = {"English", "english"}
NATIVE_LANGUAGES = {"Chinese", "chinese", "Malay", "malay", "Hindi", "hindi"}


class SpectrogramDataset(Dataset):
    """
    Shared dataset for DG experiments on audio spectrograms.

    Supports:
    - domain filtering (e.g. ["CHINESE", "MALAY"])
    - language mode filtering ("english", "native", "all")
    - optional domain labels for adversarial training
    """

    def __init__(
        self,
        annotations_file: str,
        spec_dir: str,
        domains: List[str],
        language_mode: str = "native",
        return_domain_label: bool = False,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.spec_dir = Path(spec_dir)
        self.domains = set(domains)
        self.language_mode = language_mode.lower()
        self.return_domain_label = return_domain_label

        if self.language_mode not in {"english", "native", "all"}:
            raise ValueError(
                f"language_mode must be one of ['english', 'native', 'all'], got: {language_mode}"
            )

        self.samples = self._filter_samples()

    def _language_match(self, language: str) -> bool:
        if self.language_mode == "english":
            return language in ENGLISH_LANGUAGES
        if self.language_mode == "native":
            return language in NATIVE_LANGUAGES
        return True

    def _filter_samples(self):
        samples = []

        for i in range(self.annotations.shape[0]):
            row = self.annotations.iloc[i]

            ethnicity_code = row[1].split("_")[0]   # EA / SEA / SA
            language = row.iloc[-1]

            if ethnicity_code not in ETHNICITY_TO_DOMAIN:
                continue

            domain_name = ETHNICITY_TO_DOMAIN[ethnicity_code]

            if domain_name not in self.domains:
                continue

            if not self._language_match(language):
                continue

            samples.append(row)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _get_deception_label(self, gt: str) -> torch.Tensor:
        if gt == "T":
            label = 0
        elif gt in {"F", "L"}:
            label = 1
        else:
            raise ValueError(f"Unexpected deception label: {gt}")
        return torch.tensor(label, dtype=torch.long)

    def _get_domain_label(self, ethnicity_code: str) -> torch.Tensor:
        domain_order = ["EA", "SEA", "SA"]
        return torch.tensor(domain_order.index(ethnicity_code), dtype=torch.long)

    def __getitem__(self, idx: int):
        row = self.samples[idx]

        sample_id = row[0]
        ethnicity_code = row[1].split("_")[0]
        gt = row[5]

        spectrogram_path = self.spec_dir / f"{sample_id}.pth"
        spectrogram = torch.load(spectrogram_path)

        deception_label = self._get_deception_label(gt)

        if self.return_domain_label:
            domain_label = self._get_domain_label(ethnicity_code)
            return spectrogram, deception_label, domain_label

        return spectrogram, deception_label
