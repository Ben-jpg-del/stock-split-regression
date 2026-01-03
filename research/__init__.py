# Research module for stock split data export and analysis
#
# Folder Structure:
# -----------------
# research/
# ├── data/                      # Output data files
# │   ├── price_trajectories.pkl # Training episodes (30,114 valid)
# │   └── export_metadata.json   # Export configuration and stats
# │
# ├── tools/                     # Data processing tools
# │   ├── qc_export/             # Run in QuantConnect
# │   │   ├── export_training_data.py     # Main export script
# │   │   └── read_compressed_chunks.py   # Read from Object Store
# │   │
# │   └── local/                 # Run locally
# │       └── reassemble_chunks.py        # Reassemble chunks
# │
# └── mcpt/                      # Research/analysis scripts
#     ├── bar_permute.py         # Bar permutation for Monte Carlo
#     ├── mcpt_skeleton.py       # Monte Carlo Permutation Test
#     └── research_quantbook.py  # Research notebook utilities
#
# Workflow:
# ---------
# 1. Run export_training_data.py in QuantConnect Research notebook
# 2. Run read_compressed_chunks.py to output chunks
# 3. Copy chunks to local file
# 4. Run reassemble_chunks.py to create price_trajectories.pkl
# 5. Train with: python -m RL.risk_management.train --use-qc-data
