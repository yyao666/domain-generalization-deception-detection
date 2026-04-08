import torch


def pad_spectrogram_sequence(batch):
    """
    Pad variable-length spectrograms in a batch to the same time dimension.
    """
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch,
        batch_first=True,
        padding_value=0.0,
    )
    return batch.permute(0, 2, 1)


def collate_with_labels(batch):
    """
    Collate function for:
    (spectrogram, deception_label)
    """
    spectrograms = []
    deception_labels = []

    for spec, label in batch:
        spectrograms.append(spec)
        deception_labels.append(label)

    spectrograms = pad_spectrogram_sequence(spectrograms)
    deception_labels = torch.stack(deception_labels)

    return spectrograms, deception_labels


def collate_with_domain_labels(batch):
    """
    Collate function for:
    (spectrogram, deception_label, domain_label)
    """
    spectrograms = []
    deception_labels = []
    domain_labels = []

    for spec, deception_label, domain_label in batch:
        spectrograms.append(spec)
        deception_labels.append(deception_label)
        domain_labels.append(domain_label)

    spectrograms = pad_spectrogram_sequence(spectrograms)
    deception_labels = torch.stack(deception_labels)
    domain_labels = torch.stack(domain_labels)

    return spectrograms, deception_labels, domain_labels
