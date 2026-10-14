import pandas as pd
import numpy as np

# Step 1: Load SPX data
df = pd.read_csv("spx.csv")

# Step 2: Convert date column and sort by date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Step 3: Use SPX daily total return
returns = df["totalreturn"].dropna()

# Step 4: Define Historical VaR function
def historical_var(returns, confidence_level):
    tail_probability = 1 - confidence_level
    var = -np.quantile(returns, tail_probability)
    return var

# Step 5: Calculate 95% and 99% Historical VaR
var_95 = historical_var(returns, 0.95)
var_99 = historical_var(returns, 0.99)

# Step 6: Print results
print("SPX Historical VaR - Entire Period")
print("----------------------------------")
print(f"Sample period: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Number of observations: {len(returns)}")
print(f"95% Historical VaR: {var_95:.4%}")
print(f"99% Historical VaR: {var_99:.4%}")

# Step 7: Segmented period VaR

segments = {
    "2008-2009 Financial Crisis": ("2008-03-31", "2009-12-31"),
    "2010-2019 Expansion": ("2010-01-01", "2019-12-31"),
    "2020-2021 COVID & Rebound": ("2020-01-01", "2021-12-31"),
    "2022 Rate Hike / Inflation": ("2022-01-01", "2022-12-31"),
    "2023-2025 Recent Period": ("2023-01-01", "2025-02-28")
}

print("\nSPX Historical VaR - Segmented Periods")
print("--------------------------------------")

for name, (start, end) in segments.items():
    segment_data = df[(df["date"] >= start) & (df["date"] <= end)]
    segment_returns = segment_data["totalreturn"].dropna()

    var_95_segment = historical_var(segment_returns, 0.95)
    var_99_segment = historical_var(segment_returns, 0.99)

    print(f"\n{name}")
    print(f"Period: {start} to {end}")
    print(f"Number of observations: {len(segment_returns)}")
    print(f"95% Historical VaR: {var_95_segment:.4%}")
    print(f"99% Historical VaR: {var_99_segment:.4%}")

    # Step 8: Rolling window Historical VaR

import matplotlib.pyplot as plt

window = 250

df["Rolling_HVaR_95"] = df["totalreturn"].rolling(window).apply(
    lambda x: historical_var(x, 0.95),
    raw=False
)

df["Rolling_HVaR_99"] = df["totalreturn"].rolling(window).apply(
    lambda x: historical_var(x, 0.99),
    raw=False
)

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["Rolling_HVaR_95"], label="95% Historical VaR")
plt.plot(df["date"], df["Rolling_HVaR_99"], label="99% Historical VaR")

plt.xlabel("Date")
plt.ylabel("VaR")
plt.title("250-Day Rolling Historical VaR of SPX")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("rolling_var_spx.png", dpi=300)
plt.show()
