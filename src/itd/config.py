"""Central configuration for the Insider Trading Detection (India) project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

SEED = 42
N_COMPANIES = 60
START = "2022-04-01"
END = "2025-03-31"

# Calendar quarter-ends used for SEBI "trading window closure" logic.
# Under the SEBI (Prohibition of Insider Trading) Regulations, 2015 (Schedule B),
# the trading window is closed from the end of every quarter until 48 hours
# after the declaration of financial results.
QUARTER_ENDS = [
    "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
    "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
    "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
]

# Modelling constants
MARKET_MODES = ("Market Purchase", "Market Sale")
COTRADE_WINDOW_DAYS = 5          # trading days, for co-trading / ring detection
CONTRA_WINDOW_DAYS = 180         # calendar days, SEBI contra-trade restriction (6 months)
LATE_DISCLOSURE_DAYS = 2         # trading days allowed for each disclosure hop
ESTIMATION_WINDOW = (250, 30)    # market-model estimation window: [t-250, t-30)
MIN_DISCLOSURE_VALUE = 1_000_000 # INR 10 lakh threshold (Reg 7(2))

# Ensemble weights (rank-averaged scores). LOF is still fitted and reported for comparison but gets
# weight 0: coordinated rings form dense micro-clusters, which LOF treats as *normal* ("masking").
# NOTE: this choice was made after looking at results on the synthetic data - see REPORT.md, "Limitations".
ENSEMBLE_WEIGHTS = {"iforest": 1 / 3, "lof": 0.0, "autoencoder": 1 / 3, "rules": 1 / 3}
