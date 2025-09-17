import pandas as pd
from datetime import datetime

def get_inflation_analysis(analysis_date_str: str, data_filepath: str) -> str:
    """
    Analyzes the downloaded CBS inflation data for a specific date and
    returns a human-readable text summary.
    """
    try:
        # --- Part 1: Load and Prepare the Data ---
        df = pd.read_csv(data_filepath)
        
        # The date format from CBS is "YYYYMMDD". We need to convert it.
        # This line converts '2024MM06' into a proper datetime object for the end of that month.
        df['Date'] = pd.to_datetime(df['Date'].str.replace('MM', '-'), format='%Y-%m').dt.to_period('M').dt.end_time

    except FileNotFoundError:
        return "Error: Data file not found."
    except Exception as e:
        return f"Error processing data file: {e}"

    # Convert the analysis date string to a datetime object
    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")

    # --- Part 2: Find the Correct Inflation Rate for the Date ---
    # Filter for all data points on or before the analysis date and sort to get the latest
    relevant_data = df[df['Date'] <= analysis_date].sort_values(by='Date', ascending=False)

    if relevant_data.empty:
        return f"No inflation data found on or before {analysis_date_str}."

    # Get the most recent and previous month's data
    current_inflation_row = relevant_data.iloc[0]
    current_rate = current_inflation_row['CPI_Annual_Rate']

    if len(relevant_data) < 2:
        return f"The inflation rate was {current_rate}%. Not enough prior data to determine a trend."

    previous_rate = relevant_data.iloc[1]['CPI_Annual_Rate']

    # --- Part 3: Analyze the Trend and Generate Text ---
    
    # 1. Compare to the 2% target
    target_comparison = ""
    if current_rate > 2.0:
        target_comparison = "which is above the ECB's 2% target"
    elif current_rate < 2.0:
        target_comparison = "which is below the ECB's 2% target"
    else:
        target_comparison = "which is at the ECB's 2% target"

    # 2. Check the short-term momentum
    momentum_text = ""
    if current_rate > previous_rate:
        momentum_text = f"This shows an acceleration from the previous month's {previous_rate}%."
    elif current_rate < previous_rate:
        momentum_text = f"This shows a deceleration from the previous month's {previous_rate}%."
    else:
        momentum_text = "The rate is stable compared to the previous month."

    # 3. Combine into the final summary
    final_summary = (f"Inflation Analysis for {analysis_date_str}:\n"
                     f"The annual CPI is {current_rate}%, {target_comparison}. {momentum_text}")
    
    return final_summary

# --- How to Use the Function in Your Backtest ---

# The filename of the data we downloaded in the last step
cpi_data_file = 'CPI_Raw.csv'

# --- Example 1: High and accelerating inflation ---
date_1 = "2022-08-15"
summary_1 = get_inflation_analysis(date_1, cpi_data_file)
print(summary_1)
print("-" * 30)

# --- Example 2: Negative and decelerating inflation ---
date_2 = "2023-10-15"
summary_2 = get_inflation_analysis(date_2, cpi_data_file)
print(summary_2)