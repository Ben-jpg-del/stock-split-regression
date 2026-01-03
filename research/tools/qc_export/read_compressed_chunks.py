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

# ============================================================
# CELL 3: READ STRATEGY RESULTS - NO COMPRESSION (single output)
# ============================================================
"""
from AlgorithmImports import *
import base64
import json
import pickle

qb = QuantBook()

OUTPUT_PREFIX = 'strategy_results'  # Must match run_regression_strategy.py

print("="*60)
print("READING STRATEGY RESULTS FROM OBJECT STORE")
print("="*60)

# Read metadata to get chunk count
metadata_key = f"{OUTPUT_PREFIX}_metadata.json"
try:
    metadata_str = qb.object_store.read(metadata_key)
    metadata = json.loads(metadata_str)
    num_pkl_chunks = metadata.get('num_chunks', 0)
    total_episodes = metadata.get('total_episodes', 0)
    print(f"Found metadata: {num_pkl_chunks} chunks, {total_episodes} episodes")
except Exception as e:
    print(f"Error reading metadata: {e}")
    print("Detecting chunks manually...")
    num_pkl_chunks = 0
    while True:
        try:
            key = f"{OUTPUT_PREFIX}_chunk_{num_pkl_chunks}.pkl"
            _ = qb.object_store.read_bytes(key)
            num_pkl_chunks += 1
        except:
            break
    print(f"Detected {num_pkl_chunks} chunks")
    metadata = {'num_chunks': num_pkl_chunks}

# Read and combine all pickle chunks
all_episodes = []
for i in range(num_pkl_chunks):
    key = f"{OUTPUT_PREFIX}_chunk_{i}.pkl"
    try:
        chunk_bytes = qb.object_store.read_bytes(key)
        chunk_data = pickle.loads(chunk_bytes)
        all_episodes.extend(chunk_data)
        print(f"  Loaded chunk {i}: {len(chunk_data)} episodes")
    except Exception as e:
        print(f"  Error loading chunk {i}: {e}")

print(f"\\nTotal episodes: {len(all_episodes)}")

# Serialize WITHOUT compression
pickle_bytes = pickle.dumps(all_episodes)
b64_data = base64.b64encode(pickle_bytes).decode('ascii')

print(f"Pickle size: {len(pickle_bytes):,} bytes")
print(f"Base64 size: {len(b64_data):,} characters")

# Prepare export metadata
export_metadata = {
    'total_episodes': len(all_episodes),
    'compressed': False,  # NOT compressed
    'total_b64_size': len(b64_data),
    'num_chunks': 1,
    'original_metadata': metadata,
}

# Print everything in one go
print("\\n---METADATA_START---")
print(json.dumps(export_metadata))
print("---METADATA_END---")

print("\\n---CHUNK_1_START---")
print(b64_data)
print("---CHUNK_1_END---")

print("\\n" + "="*60)
print("DONE - Copy everything above to data.txt")
print("="*60)
"""

# ============================================================
# CELL 4: (Not needed - Cell 3 outputs everything at once)
# ============================================================
