# Kronos Trading

Automated trading system using [Kronos](https://github.com/shiyu-coder/Kronos) -- a foundation model for financial candlestick (K-line) forecasting.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Test

```bash
python test_kronos.py
```

## Models

Models are downloaded automatically from HuggingFace on first run:

| Model | Params | Context |
|---|---|---|
| Kronos-mini | 4.1M | 2048 |
| Kronos-small | 24.7M | 512 |
| Kronos-base | 102.3M | 512 |

## Credits

- Kronos model: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) (MIT License)
- Paper: [arXiv:2508.02739](https://arxiv.org/abs/2508.02739) (AAAI 2026)
