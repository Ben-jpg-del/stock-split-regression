"""
Reassemble Data Chunks from QuantConnect Export

Run this locally after copying chunks from the QuantConnect notebook output.

Usage:
    python research/tools/local/reassemble_chunks.py

You can either:
    1. Paste chunks interactively when prompted
    2. Save all output to a text file and provide the file path
"""

import base64
import pickle
import json
import re
import sys
import zlib
from pathlib import Path


def extract_from_file(filepath: str) -> tuple:
    """Extract metadata and chunks from a saved output file."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract metadata
    meta_match = re.search(r'---METADATA_START---\s*(.+?)\s*---METADATA_END---', content, re.DOTALL)
    if not meta_match:
        raise ValueError("Could not find metadata in file")
    metadata = json.loads(meta_match.group(1).strip())

    # Extract chunks
    chunks = []
    chunk_pattern = r'---CHUNK_(\d+)_START---\s*(.+?)\s*---CHUNK_\1_END---'
    for match in re.finditer(chunk_pattern, content, re.DOTALL):
        chunk_num = int(match.group(1))
        chunk_data = match.group(2).strip()
        chunks.append((chunk_num, chunk_data))

    # Sort by chunk number
    chunks.sort(key=lambda x: x[0])

    return metadata, [c[1] for c in chunks]


def interactive_input() -> tuple:
    """Get metadata and chunks via interactive input."""
    print("\n" + "="*60)
    print("INTERACTIVE CHUNK INPUT")
    print("="*60)

    # Get metadata
    print("\nPaste the METADATA (everything between ---METADATA_START--- and ---METADATA_END---):")
    print("(Press Enter twice when done)")

    lines = []
    while True:
        line = input()
        if line == "":
            if lines:
                break
        else:
            lines.append(line)

    metadata_str = "\n".join(lines).strip()
    # Clean up markers if included
    metadata_str = re.sub(r'---METADATA_START---', '', metadata_str)
    metadata_str = re.sub(r'---METADATA_END---', '', metadata_str)
    metadata = json.loads(metadata_str.strip())

    num_chunks = metadata['num_chunks']
    print(f"\nExpecting {num_chunks} chunks...")

    chunks = []
    for i in range(1, num_chunks + 1):
        print(f"\nPaste CHUNK {i}/{num_chunks}:")
        print(f"(Everything between ---CHUNK_{i}_START--- and ---CHUNK_{i}_END---)")
        print("(Press Enter twice when done)")
        lines = []
        while True:
            line = input()
            if line == "":
                if lines:
                    break
            else:
                lines.append(line)

        chunk_str = "\n".join(lines).strip()
        # Clean up markers if included
        chunk_str = re.sub(r'---CHUNK_\d+_START---', '', chunk_str)
        chunk_str = re.sub(r'---CHUNK_\d+_END---', '', chunk_str)
        chunks.append(chunk_str.strip())
        print(f"  Chunk {i} received: {len(chunks[-1])} characters")

    return metadata, chunks


def reassemble(metadata: dict, chunks: list, output_path: str = 'research/data/price_trajectories.pkl'):
    """Reassemble chunks into pickle file."""
    print("\n" + "="*60)
    print("REASSEMBLING DATA")
    print("="*60)

    # Combine chunks
    b64_data = "".join(chunks)
    print(f"Total base64 data: {len(b64_data)} characters")

    expected_size = metadata.get('total_b64_size', 0)
    if expected_size and len(b64_data) != expected_size:
        print(f"WARNING: Expected {expected_size} chars, got {len(b64_data)}")
        diff = expected_size - len(b64_data)
        if diff > 0:
            print(f"  Missing {diff} characters - some chunks may be incomplete")
        else:
            print(f"  Extra {-diff} characters - may have duplicated data")

    # Decode base64
    try:
        decoded_bytes = base64.b64decode(b64_data)
        print(f"Decoded base64: {len(decoded_bytes)} bytes")
    except Exception as e:
        print(f"ERROR decoding base64: {e}")
        print("Check that all chunks were copied correctly")
        return False

    # Decompress if compressed
    if metadata.get('compressed', False):
        try:
            pickle_bytes = zlib.decompress(decoded_bytes)
            print(f"Decompressed: {len(pickle_bytes)} bytes")
        except Exception as e:
            print(f"ERROR decompressing: {e}")
            return False
    else:
        pickle_bytes = decoded_bytes

    # Load pickle to verify
    try:
        episodes = pickle.loads(pickle_bytes)
        print(f"Loaded {len(episodes)} episodes")
    except Exception as e:
        print(f"ERROR loading pickle: {e}")
        return False

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(episodes, f)

    print(f"\nSaved to: {output_file}")

    # Also save metadata
    metadata_file = output_file.parent / 'export_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Total episodes: {len(episodes)}")

    if episodes:
        long_count = sum(1 for e in episodes if e['direction'] > 0)
        short_count = sum(1 for e in episodes if e['direction'] < 0)
        penny_count = sum(1 for e in episodes if e['is_penny_stock'])
        symbols = set(e['symbol'] for e in episodes)

        print(f"Unique symbols: {len(symbols)}")
        print(f"Long episodes: {long_count}")
        print(f"Short episodes: {short_count}")
        print(f"Penny stock episodes: {penny_count}")

        pnls = [e['final_pnl_pct'] for e in episodes]
        import numpy as np
        print(f"P&L - Mean: {np.mean(pnls):.2%}, Std: {np.std(pnls):.2%}")
        print(f"P&L - Min: {np.min(pnls):.2%}, Max: {np.max(pnls):.2%}")

    print("\nYou can now train with:")
    print("  python -m RL.risk_management.train --use-qc-data")

    return True


def main():
    print("="*60)
    print("QuantConnect Data Chunk Reassembler")
    print("="*60)

    # Check if file provided as argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\nReading from file: {filepath}")
        try:
            metadata, chunks = extract_from_file(filepath)
            print(f"Found {len(chunks)} chunks in file")
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        # Check for optional output path argument
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
        else:
            output_path = "research/data/price_trajectories.pkl"

        reassemble(metadata, chunks, output_path)
    else:
        print("\nOptions:")
        print("  1. Paste chunks interactively")
        print("  2. Provide path to saved output file")

        choice = input("\nEnter choice (1 or 2), or file path: ").strip()

        if choice == "1":
            metadata, chunks = interactive_input()
        elif choice == "2":
            filepath = input("Enter file path: ").strip()
            metadata, chunks = extract_from_file(filepath)
        elif Path(choice).exists():
            metadata, chunks = extract_from_file(choice)
        else:
            print("Invalid choice")
            return

        # Reassemble
        output_path = input("\nOutput path [research/data/price_trajectories.pkl]: ").strip()
        if not output_path:
            output_path = "research/data/price_trajectories.pkl"

        reassemble(metadata, chunks, output_path)


if __name__ == "__main__":
    main()
