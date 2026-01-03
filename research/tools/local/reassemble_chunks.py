"""
Reassemble Data Chunks from QuantConnect Export

Run this locally after downloading chunks from QuantConnect Object Store.

Usage:
    # For Object Store pickle files (recommended):
    python research/tools/local/reassemble_chunks.py --object-store ./downloads/

    # For text-based chunks (legacy):
    python research/tools/local/reassemble_chunks.py notebook_output.txt

Options:
    --object-store DIR    Directory containing downloaded .pkl chunk files
    FILE                  Text file with marker-based chunks (legacy)
"""

import base64
import pickle
import json
import re
import sys
import zlib
import glob
from pathlib import Path


def reassemble_object_store(directory: str, output_path: str = 'research/data/price_trajectories.pkl'):
    """
    Reassemble pickle chunks downloaded from QuantConnect Object Store.

    Args:
        directory: Directory containing strategy_results_chunk_N.pkl files
        output_path: Where to save the combined file
    """
    print("\n" + "="*60)
    print("REASSEMBLING OBJECT STORE CHUNKS")
    print("="*60)

    dir_path = Path(directory)

    # Find all chunk files
    chunk_pattern = str(dir_path / "*_chunk_*.pkl")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        # Try alternate pattern
        chunk_pattern = str(dir_path / "*.pkl")
        chunk_files = [f for f in glob.glob(chunk_pattern) if 'metadata' not in f.lower()]
        chunk_files = sorted(chunk_files)

    if not chunk_files:
        print(f"ERROR: No chunk files found in {directory}")
        print(f"Looking for pattern: *_chunk_*.pkl or *.pkl")
        return False

    print(f"Found {len(chunk_files)} chunk files:")
    for f in chunk_files:
        print(f"  - {Path(f).name}")

    # Load and combine all chunks
    all_episodes = []
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            all_episodes.extend(chunk_data)
            print(f"  Loaded {len(chunk_data)} episodes from {Path(chunk_file).name}")
        except Exception as e:
            print(f"  ERROR loading {chunk_file}: {e}")
            return False

    print(f"\nTotal episodes combined: {len(all_episodes)}")

    # Load metadata if available
    metadata_files = list(dir_path.glob("*metadata*.json"))
    metadata = {}
    if metadata_files:
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata from {metadata_files[0].name}")
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")

    # Save combined file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    print(f"\nSaved combined data to: {output_file}")

    # Save metadata
    if metadata:
        metadata['total_episodes'] = len(all_episodes)
        metadata_out = output_file.parent / 'export_metadata.json'
        with open(metadata_out, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_out}")

    # Print summary
    print_episode_summary(all_episodes)

    return True


def print_episode_summary(episodes):
    """Print summary statistics for episodes."""
    import numpy as np

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    if not episodes:
        print("No episodes!")
        return

    print(f"Total episodes: {len(episodes)}")

    symbols = set(e.get('symbol', 'unknown') for e in episodes)
    print(f"Unique symbols: {len(symbols)}")

    long_count = sum(1 for e in episodes if e.get('direction', 0) > 0)
    short_count = sum(1 for e in episodes if e.get('direction', 0) < 0)
    print(f"Long episodes: {long_count}")
    print(f"Short episodes: {short_count}")

    penny_count = sum(1 for e in episodes if e.get('is_penny_stock', False))
    print(f"Penny stock episodes: {penny_count}")

    # P&L statistics
    pnl_key = 'final_pnl_pct' if 'final_pnl_pct' in episodes[0] else 'actual_return'
    pnls = [e.get(pnl_key, 0) for e in episodes]
    print(f"\nP&L Statistics ({pnl_key}):")
    print(f"  Mean: {np.mean(pnls):.4f} ({np.mean(pnls)*100:.2f}%)")
    print(f"  Std:  {np.std(pnls):.4f}")
    print(f"  Min:  {np.min(pnls):.4f}")
    print(f"  Max:  {np.max(pnls):.4f}")

    # Check for prediction data
    if 'predicted_return' in episodes[0]:
        preds = [e.get('predicted_return', 0) for e in episodes]
        corr = np.corrcoef(preds, pnls)[0, 1] if len(preds) > 1 else 0
        print(f"\nPrediction Quality:")
        print(f"  Prediction correlation: {corr:.4f}")
        print(f"  Avg predicted: {np.mean(preds):.4f}")

    print("\nReady for training:")
    print("  python -m RL.risk_management.train --use-qc-data")


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

    # Check for --object-store flag
    if len(sys.argv) > 1 and sys.argv[1] == '--object-store':
        if len(sys.argv) < 3:
            print("ERROR: --object-store requires a directory path")
            print("Usage: python reassemble_chunks.py --object-store ./downloads/")
            return

        directory = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else "research/data/price_trajectories.pkl"

        reassemble_object_store(directory, output_path)
        return

    # Check if file provided as argument (legacy mode)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

        # Check if it's a directory (Object Store mode)
        if Path(filepath).is_dir():
            output_path = sys.argv[2] if len(sys.argv) > 2 else "research/data/price_trajectories.pkl"
            reassemble_object_store(filepath, output_path)
            return

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
        print("  1. Reassemble Object Store downloads (recommended)")
        print("  2. Paste chunks interactively (legacy)")
        print("  3. Provide path to saved text output file (legacy)")

        choice = input("\nEnter choice (1, 2, or 3): ").strip()

        if choice == "1":
            directory = input("Enter directory containing .pkl chunk files: ").strip()
            output_path = input("Output path [research/data/price_trajectories.pkl]: ").strip()
            if not output_path:
                output_path = "research/data/price_trajectories.pkl"
            reassemble_object_store(directory, output_path)
        elif choice == "2":
            metadata, chunks = interactive_input()
            output_path = input("\nOutput path [research/data/price_trajectories.pkl]: ").strip()
            if not output_path:
                output_path = "research/data/price_trajectories.pkl"
            reassemble(metadata, chunks, output_path)
        elif choice == "3":
            filepath = input("Enter file path: ").strip()
            try:
                metadata, chunks = extract_from_file(filepath)
            except Exception as e:
                print(f"Error: {e}")
                return
            output_path = input("\nOutput path [research/data/price_trajectories.pkl]: ").strip()
            if not output_path:
                output_path = "research/data/price_trajectories.pkl"
            reassemble(metadata, chunks, output_path)
        else:
            print("Invalid choice")
            return


if __name__ == "__main__":
    main()
