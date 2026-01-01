# Stock Split Trading Algorithm

A machine learning-based trading strategy that predicts returns around stock split events using sector momentum and split factors.


## Strategy Results


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
