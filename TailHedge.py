import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. Load Market Data
# ==================================================

# Load S&P 500 index data
spx = pd.read_csv("spx.csv")

spx["date"] = pd.to_datetime(spx["date"])


# Load SPX option daily price data
option = pd.read_csv(
    "spx_selected_option_daily_price_202502.csv"
)


# Check data
print("SPX data:")
print(spx.head())

print("\nOption data:")
print(option.head())

print("\nOption columns:")
print(option.columns)

print("\nOption shape:")
print(option.shape)



# ==================================================
# 2. Tail Risk Measurement
# ==================================================

# Calculate daily return
spx["daily_return"] = spx["closeprice"].pct_change()


# Historical VaR (95%)
VaR_95 = spx["daily_return"].quantile(0.05)

print("\n95% Historical VaR:", VaR_95)



# Plot return distribution

plt.figure(figsize=(8,5))

plt.hist(
    spx["daily_return"].dropna(),
    bins=100
)

plt.axvline(
    VaR_95,
    linestyle="--",
    label=f"95% VaR={VaR_95:.2%}"
)


plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.title("S&P 500 Daily Return Distribution")

plt.legend()

plt.show()



# ==================================================
# 3. Tail Hedge Strategy
# ==================================================



# Check option data structure
print("\nOption data information:")
print(option.info())


print("\nFirst 5 rows of option data:")
print(option.head())
print(spx.columns)

# ==================================================
# 3. Tail Hedge Strategy
# ==================================================

# Convert dates

option["date"] = pd.to_datetime(
    option["date"].str.strip(),
    errors="coerce"
)

print("Missing option dates:")
print(option["date"].isna().sum())


# Remove invalid dates
option = option.dropna(subset=["date"])


# Merge

data = pd.merge(
    spx,
    option,
    on="date",
    how="inner"
)


print("\nMerged data:")
print(data.head())

print("\nMerged shape:")
print(data.shape)

# ==================================================
# 4. Portfolio Initialization
# ==================================================

def initialize_portfolio(first_day):

    initial_cash = 100


    # SPX price
    spx_price = first_day["closeprice"]


    # Each option leg uses 0.5% portfolio value
    option_budget = initial_cash * 0.005


    # Buy three put options
    leg1_unit = option_budget / first_day["leg_1_ask"]
    leg2_unit = option_budget / first_day["leg_2_ask"]
    leg3_unit = option_budget / first_day["leg_3_ask"]


    # Total option cost
    option_cost = (
        leg1_unit * first_day["leg_1_ask"]
        +
        leg2_unit * first_day["leg_2_ask"]
        +
        leg3_unit * first_day["leg_3_ask"]
    )


    # Remaining money buys SPX
    spx_unit = (
        initial_cash - option_cost
    ) / spx_price


    portfolio = {

        "spx_unit": spx_unit,

        "leg1_unit": leg1_unit,

        "leg2_unit": leg2_unit,

        "leg3_unit": leg3_unit

    }


    return portfolio



# Initialize using first trading day

portfolio = initialize_portfolio(data.iloc[0])


print("\nInitial portfolio:")
print(portfolio)

# ==================================================
# Portfolio Value Function
# ==================================================

def calculate_portfolio_value(row, portfolio):

    spx_value = (
        row["closeprice"]
        *
        portfolio["spx_unit"]
    )


    option_value = (
        row["leg_1_ask"]
        *
        portfolio["leg1_unit"]
        +
        row["leg_2_ask"]
        *
        portfolio["leg2_unit"]
        +
        row["leg_3_ask"]
        *
        portfolio["leg3_unit"]
    )


    portfolio_value = (
        spx_value
        +
        option_value
    )


    return portfolio_value

# ==================================================
# Rolling Function
# ==================================================

def get_roll_dates(data):

    roll_dates = []

    temp = data.copy()

    temp["month"] = temp["date"].dt.month
    temp["year"] = temp["date"].dt.year


    # quarterly months
    months = [3, 6, 9, 12]


    for year in temp["year"].unique():

        for month in months:

            subset = temp[
                (temp["year"] == year)
                &
                (temp["month"] == month)
            ]


            if len(subset) == 0:
                continue


            # third Friday
            fridays = subset[
                subset["date"].dt.weekday == 4
            ]


            if len(fridays) >= 3:

                third_friday = fridays.iloc[2]["date"]


                # previous business day
                previous_day = subset[
                    subset["date"] < third_friday
                ].iloc[-1]["date"]


                roll_dates.append(previous_day)


    return roll_dates



# ==================================================
# Option Rolling Function
# ==================================================

def roll_option(row, portfolio_value, portfolio):

    option_budget = portfolio_value * 0.005


    portfolio["leg1_unit"] = (
        option_budget / row["leg_1_ask"]
    )

    portfolio["leg2_unit"] = (
        option_budget / row["leg_2_ask"]
    )

    portfolio["leg3_unit"] = (
        option_budget / row["leg_3_ask"]
    )


    return portfolio


# Test initial portfolio value

initial_value = calculate_portfolio_value(
    data.iloc[0],
    portfolio
)

print("\nInitial portfolio value:")
print(initial_value)

# ==================================================
# 5. Daily Portfolio Backtest with Rolling
# ==================================================

# Generate rolling dates
roll_dates = get_roll_dates(data)

print("Number of roll dates:")
print(len(roll_dates))

portfolio_values = []


# Copy initial portfolio
current_portfolio = portfolio.copy()


# Initial portfolio value
current_value = 100



for _, row in data.iterrows():


    # Roll option positions
    if row["date"] in roll_dates:

        current_portfolio = roll_option(
            row,
            current_value,
            current_portfolio
        )


    # Calculate daily portfolio value

    current_value = calculate_portfolio_value(
        row,
        current_portfolio
    )


    portfolio_values.append(current_value)



data["portfolio_value"] = portfolio_values



print(data[
    [
        "date",
        "closeprice",
        "portfolio_value"
    ]
].head())



plt.figure(figsize=(10,5))


plt.plot(
    data["date"],
    data["portfolio_value"],
    label="Rolling Tail Hedge Portfolio"
)


plt.xlabel("Date")
plt.ylabel("Portfolio Value")


plt.title(
    "Rolling Tail Hedge Portfolio Performance"
)


plt.legend()

plt.show()

# ==================================================
# 6. SPX Benchmark
# ==================================================

initial_spx_price = data.iloc[0]["closeprice"]


data["spx_only_value"] = (
    100
    *
    data["closeprice"]
    /
    initial_spx_price
)

# ==================================================
# 7. Performance Plot
# ==================================================

plt.figure(figsize=(10,5))


plt.plot(
    data["date"],
    data["spx_only_value"],
    label="SPX Buy & Hold"
)


plt.plot(
    data["date"],
    data["portfolio_value"],
    label="Tail Hedge Portfolio"
)


plt.xlabel("Date")
plt.ylabel("Portfolio Value")

plt.title(
    "SPX vs Tail Hedge Portfolio Performance"
)


plt.legend()

plt.show()

# ==================================================
# 8. Performance Metrics
# ==================================================

def performance_metrics(series):

    total_return = (
        series.iloc[-1]
        /
        series.iloc[0]
        -
        1
    )


    daily_return = series.pct_change()


    volatility = (
        daily_return.std()
        *
        (252 ** 0.5)
    )


    running_max = series.cummax()

    drawdown = (
        series-running_max
    ) / running_max


    max_drawdown = drawdown.min()


    return (
        total_return,
        volatility,
        max_drawdown
    )



spx_metrics = performance_metrics(
    data["spx_only_value"]
)


hedge_metrics = performance_metrics(
    data["portfolio_value"]
)


print("\nSPX:")
print(spx_metrics)


print("\nTail Hedge:")
print(hedge_metrics)



# To be implemented:
# 1. Initialize portfolio
# 2. Option sizing
# 3. Option rolling
# 4. Portfolio value calculation

