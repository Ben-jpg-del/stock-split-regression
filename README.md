# Stock Split Trading Algorithm

A machine learning-based trading strategy that predicts returns around stock split events using sector momentum and split factors.


## Strategy Results
<img width="1466" height="810" alt="Screenshot 2025-12-31 202115" src="https://github.com/user-attachments/assets/cc37fa99-71be-4cf2-b1e0-5b8e571ab553" /> 
<img width="944" height="328" alt="Screenshot 2025-12-31 212233" src="https://github.com/user-attachments/assets/a1e5e276-a62b-4808-8949-ff717302ce81" />
<img width="1050" height="360" alt="Screenshot 2025-12-31 212215" src="https://github.com/user-attachments/assets/acdee32c-8de3-4320-9a12-f6c7c96a491b" />

## Repository Structure

```
mcpt-main/
├── quantconnect/           # QuantConnect algorithm
│   └── main.py             # Main algorithm for QC platform
├── standalone/             # Standalone paper trading
│   └── paper_trading.py    # Alpaca + Polygon.io implementation
├── research/               # Research and analysis
│   ├── research_quantbook.py
│   ├── bar_permute.py
│   └── mcpt_skeleton.py
├── logs/                   # Log files (gitignored)
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
└── README.md
```
