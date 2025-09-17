import pandas as pd
from datetime import datetime

def get_interest_rate_analysis(analysis_date_str: str, data_filepath: str) -> str:
    """
    Analyzes ECB interest rate data for a specific date and returns a text summary.

    Args:
        analysis_date_str: The date you are analyzing, as a string (e.g., "2023-06-15").
        data_filepath: The path to your downloaded ECB CSV file.

    Returns:
        A human-readable text summary of the interest rate situation.
    """
    # --- Part 1: Load and Prepare the Data ---
    try:
        # Load the CSV file into a pandas DataFrame.
        df = pd.read_csv(data_filepath)

        # Rename columns for easier access. The names are long and complex.
        df.columns = ['DATE', 'TIME_PERIOD', 'RATE']

        # Convert the 'TIME_PERIOD' column to proper datetime objects for comparison.
        # 'coerce' will turn any invalid date formats into 'NaT' (Not a Time).
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')
        df.dropna(subset=['TIME_PERIOD'], inplace=True) # Remove any rows that failed conversion.

    except FileNotFoundError:
        return "Error: Data file not found."
    except Exception as e:
        return f"Error processing data file: {e}"

    # Convert the analysis date string to a datetime object.
    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")


    # --- Part 2: Find the Correct Rate for the Analysis Date ---
    
    # Filter the DataFrame to find all rate changes that happened ON or BEFORE our analysis date.
    relevant_data = df[df['TIME_PERIOD'] <= analysis_date].sort_values(by='TIME_PERIOD', ascending=False)

    if relevant_data.empty:
        return f"No historical interest rate data found on or before {analysis_date_str}."

    # The most recent rate change is the first one in our sorted list.
    current_rate_row = relevant_data.iloc[0]
    current_rate = current_rate_row['RATE']

    # We need at least two data points to determine a trend.
    if len(relevant_data) < 2:
        return f"The ECB interest rate on {analysis_date_str} was {current_rate}%. Not enough prior data to determine a trend."

    # The previous rate change is the second one in our list.
    previous_rate_row = relevant_data.iloc[1]
    previous_rate = previous_rate_row['RATE']
    
    
    # --- Part 3: Analyze the Trend and Generate Text ---

    trend_text = ""
    if current_rate > previous_rate:
        change = round(current_rate - previous_rate, 2)
        trend_text = f"This reflects a recent hike of {change}%, signaling a hawkish policy stance."
    elif current_rate < previous_rate:
        change = round(previous_rate - current_rate, 2)
        trend_text = f"This reflects a recent cut of {change}%, signaling a dovish policy stance."
    else:
        trend_text = "The rate was held unchanged from the previous decision, suggesting a neutral or 'wait-and-see' approach."

    # Combine everything into a final summary.
    final_summary = f"ECB Interest Rate Analysis for {analysis_date_str}:\nThe key rate is {current_rate}%. {trend_text}"
    
    return final_summary

# --- Step 3: Use the Function ---

# This is how you would use the function in your main script.
# Make sure the CSV file is in the same folder as your Python script.
csv_file = 'ECB_Data.csv' 

# --- Example 1: A date during a period of no change ---
date_to_analyze_1 = "2024-03-20"
summary1 = get_interest_rate_analysis(date_to_analyze_1, csv_file)
print(summary1)
print("-" * 30)

# --- Example 2: A date right after a rate hike ---
date_to_analyze_2 = "2023-09-21" 
summary2 = get_interest_rate_analysis(date_to_analyze_2, csv_file)
print(summary2)
print("-" * 30)

# --- Example 3: A date right after a rate cut ---
date_to_analyze_3 = "2019-09-19"
summary3 = get_interest_rate_analysis(date_to_analyze_3, csv_file)
print(summary3)