# -*- coding: utf-8 -*-
"""Block reader for Traces.dat (Assignment 10):
Layout: idrandom | 1000*samples (int8) | idrandom | 1000*samples | ...
"""
import numpy as np

def iter_blocks(path, points_per_trace=1000, block_size=2000):
    elts_per_trace = 1 + points_per_trace
    dt = np.dtype('i1')  # int8 is fine for id as it's 0/1
    total = None
    with open(path, 'rb') as f:
        # Infer total by file size
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        assert size % elts_per_trace == 0, "File size is not a multiple of trace length"
        total = size // elts_per_trace
        # Read in blocks
        remaining = total
        while remaining > 0:
            take = int(min(block_size, remaining))
            buf = np.frombuffer(f.read(take * elts_per_trace), dtype=dt)
            A = buf.reshape(take, elts_per_trace)
            ids = A[:, 0].astype(np.uint8)
            traces = A[:, 1:].astype(np.int8)
            yield traces, ids
            remaining -= take
