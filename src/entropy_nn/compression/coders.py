from dataclasses import dataclass
from typing import Literal

import constriction
import numpy as np
import torch


@dataclass
class EncodedTensor:
    """
    Encoded representation of a quantized tensor.

      - coder_kind: "ans" or "huffman"
      - probs: categorical probabilities over alphabet {0..K-1}
      - qmax: original symmetric quantizer max magnitude (for un-shifting)
      - scale: dequant scale
      - shape: original tensor shape
      - n: number of symbols
      - payload: compressed data
    """

    coder_kind: Literal["huffman", "ans"]
    probs: np.ndarray
    qmax: int
    scale: float
    shape: tuple[int, ...]
    n: int
    payload: np.ndarray


@staticmethod
def hist_probs(symbols_shifted: torch.Tensor, K: int) -> np.ndarray:
    # returns numpy float32 probs

    # torch.bincount expects nonnegative integer input
    counts = torch.bincount(symbols_shifted.to(torch.int64), minlength=K).to(torch.float64)
    counts = counts + 1.0  # Laplace smoothing to avoid zero-prob symbols
    probs = (counts / counts.sum()).cpu().numpy().astype(np.float32)
    return probs


@staticmethod
def encode_tensor(q: torch.Tensor, scale: float, qmax: int, *, coder_type: Literal["huffman", "ans"]) -> EncodedTensor:
    q_int = q.detach().to(torch.int32).reshape(-1).cpu()
    s = (q_int + qmax).to(torch.int32)  # symbols in {0..2*qmax}
    s_np = s.numpy().astype(np.int32)

    K = 2 * qmax + 1
    probs = hist_probs(s, K=K)

    payload = None

    if coder_type == "ans":
        model = constriction.stream.model.Categorical(probs, perfect=False)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(s_np, model)  # reverse for stack semantics
        payload = coder.get_compressed()

    elif coder_type == "huffman":
        coder = constriction.symbol.StackCoder()
        encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probs)
        for sym in s_np[::-1]:  # reverse for stack semantics
            coder.encode_symbol(int(sym), encoder_codebook)
        payload, _bitrate = coder.get_compressed()

    else:
        raise ValueError(f"Unsupported coder: {coder_type}")

    return EncodedTensor(
        coder_kind=coder_type,
        probs=probs,
        qmax=int(qmax),
        scale=float(scale),
        shape=tuple(q.shape),
        n=int(s_np.size),
        payload=payload,
    )


@staticmethod
def decode_tensor(encoded: EncodedTensor) -> torch.Tensor:
    if encoded.coder_kind == "ans":
        model = constriction.stream.model.Categorical(encoded.probs, perfect=False)
        ans = constriction.stream.stack.AnsCoder(encoded.payload)
        s = ans.decode(model, encoded.n).astype(np.int32)

    elif encoded.coder_kind == "huffman":
        decoder = constriction.symbol.StackCoder(encoded.payload)
        decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(encoded.probs)

        s = np.empty(encoded.n, dtype=np.int32)
        for i in range(encoded.n):
            s[i] = int(decoder.decode_symbol(decoder_codebook))

    else:
        raise ValueError(f"Unsupported coder: {encoded.coder_kind}")

    q = s - encoded.qmax
    out = torch.from_numpy(q).to(torch.float32).mul_(encoded.scale)
    return out.reshape(encoded.shape)
