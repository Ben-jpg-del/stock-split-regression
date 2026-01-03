"""
Read from Object Store - COMPRESSED VERSION

Compresses data before base64 encoding to dramatically reduce chunk count.
Pickle data typically compresses 5-10x smaller.

CELL 1: Run this first to get metadata and chunk count
CELL 2: Run repeatedly for each chunk
"""

# ============================================================
# CELL 1: INITIALIZE WITH COMPRESSION (paste in first cell)
# ============================================================
"""
from AlgorithmImports import *
import base64
import json
import zlib

qb = QuantBook()

PICKLE_KEY = 'price_trajectories.pkl'
CHUNK_SIZE = 50000  # Larger chunks since compressed

print("Reading from Object Store...")
pickle_buffer = qb.object_store.read_bytes(PICKLE_KEY)
pickle_bytes = bytes(pickle_buffer)
print(f"Original size: {len(pickle_bytes):,} bytes")

# Compress with zlib (level 9 = max compression)
compressed = zlib.compress(pickle_bytes, level=9)
print(f"Compressed size: {len(compressed):,} bytes ({len(compressed)/len(pickle_bytes):.1%})")

# Base64 encode compressed data
b64_data = base64.b64encode(compressed).decode('ascii')
total_size = len(b64_data)
num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"Base64 size: {total_size:,} characters")
print(f"Number of chunks: {num_chunks}")

# Read metadata
metadata_str = qb.object_store.read('export_metadata.json')
metadata = json.loads(metadata_str)
metadata['num_chunks'] = num_chunks
metadata['total_b64_size'] = total_size
metadata['compressed'] = True
metadata['original_size'] = len(pickle_bytes)
metadata['compressed_size'] = len(compressed)

print("\\n---METADATA_START---")
print(json.dumps(metadata))
print("---METADATA_END---")

print(f"\\nRun Cell 2 with CHUNK_NUMBER = 1, 2, 3, ... {num_chunks}")
"""

# ============================================================
# CELL 2: GET ONE CHUNK (paste in second cell, run repeatedly)
# ============================================================
"""
CHUNK_NUMBER = 1  # <-- CHANGE THIS: 1, 2, 3, ... up to num_chunks

start = (CHUNK_NUMBER - 1) * CHUNK_SIZE
end = min(CHUNK_NUMBER * CHUNK_SIZE, total_size)
chunk = b64_data[start:end]

print(f"---CHUNK_{CHUNK_NUMBER}_START---")
print(chunk)
print(f"---CHUNK_{CHUNK_NUMBER}_END---")
"""
