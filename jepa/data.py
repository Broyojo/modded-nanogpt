import glob
from pathlib import Path

import torch
from torch import Tensor

BOS_ID = 50256


def load_shard(file: Path) -> Tensor:
    """Load tokens from a fineweb-style .bin file (uint16). Returns int64 tensor on CPU."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert int(header[0]) == 20240520, f"magic mismatch in {file}"
    assert int(header[1]) == 1
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens.to(torch.int64)


def find_bos_starts(tokens: Tensor, seq_len: int) -> Tensor:
    """Indices of BOS tokens that have at least `seq_len` tokens after them (inclusive of BOS)."""
    bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0]
    return bos_idx[bos_idx + seq_len <= len(tokens)]


class FineWebBatcher:
    """
    Streaming (B, T) int64 batches of FineWeb tokens. Each row starts at a BOS.
    Per-rank striding so different ranks see different samples.
    Wraps around the shard list at end (multi-epoch).
    """

    def __init__(
        self,
        glob_pattern: str,
        seqs_per_step: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str = "cuda",
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.files = sorted([Path(p) for p in glob.glob(glob_pattern)])
        if not self.files:
            raise FileNotFoundError(f"no files matched: {glob_pattern}")
        self.seqs_per_step = seqs_per_step
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.gen = torch.Generator().manual_seed(seed + rank * 1009)
        self._file_idx = 0
        self._load_current_shard()

    def _load_current_shard(self):
        path = self.files[self._file_idx % len(self.files)]
        self.tokens = load_shard(path)
        self.bos_idx = find_bos_starts(self.tokens, self.seq_len)
        if self.shuffle:
            perm = torch.randperm(len(self.bos_idx), generator=self.gen)
            self.bos_idx = self.bos_idx[perm]
        self.local_bos = self.bos_idx[self.rank::self.world_size]
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tensor:
        if self.cursor + self.seqs_per_step > len(self.local_bos):
            self._file_idx += 1
            self._load_current_shard()
        starts = self.local_bos[self.cursor:self.cursor + self.seqs_per_step]
        self.cursor += self.seqs_per_step
        out = torch.stack([self.tokens[s:s + self.seq_len] for s in starts])
        return out.to(self.device, non_blocking=True)


def synthetic_batch(B: int, T: int, vocab_size: int, device: torch.device | str = "cuda", seed: int | None = None) -> Tensor:
    """Random tokens for tests. Each row starts with BOS so it mirrors real data structure."""
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    out = torch.randint(0, vocab_size, (B, T), device=device, dtype=torch.int64, generator=g)
    out[:, 0] = BOS_ID
    return out
