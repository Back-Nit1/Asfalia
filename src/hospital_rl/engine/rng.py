"""Keyed random streams: stateless draws so changing one branch does not shift others."""
import hashlib


def draw(seed: int, stream: str, *key) -> float:
    h = hashlib.blake2b(repr((seed, stream) + key).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2 ** 64
