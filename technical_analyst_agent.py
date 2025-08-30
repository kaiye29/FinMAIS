import pandas as pd
import pandas_ta as ta
from datetime import datetime

def generate_technical_prompt(analysis_date_str: str, data_filepath: str, company_ticker: str) -> str:
    """
    Reads a clean, tidy stock price CSV, calculates technical indicators for a
    specific company, and generates a prompt for the Technical Analyst Agent.
    """
    try:
        # --- Part 1: Load the Clean Data ---
        print(f"Loading clean data from: {data_filepath}")
        df = pd.read_csv(data_filepath)

        # Convert the 'Date' column to proper datetime objects
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

    except FileNotFoundError:
        return f"Error: The file '{data_filepath}' was not found. Please ensure it's in the same folder and named correctly."
    except KeyError as e:
        return f"KeyError: Could not find the column {e}. Please ensure your CSV has these exact columns: Date,Ticker,Open,High,Low,Close,Volume"
    except Exception as e:
        return f"An error occurred: {e}"

    # --- Part 2: Isolate and Analyze a Single Company ---
    print(f"Extracting and analyzing data for {company_ticker}...")
    company_df = df[df['Ticker'] == company_ticker].copy()
    company_df.set_index('Date', inplace=True)

    if company_df.empty:
        return f"Error: No data found for ticker '{company_ticker}' in the file."

    # --- Part 3: Calculate Technical Indicators ---
    print("Calculating technical indicators...")
    company_df.ta.sma(length=50, append=True)
    company_df.ta.sma(length=200, append=True)
    company_df.ta.rsi(length=14, append=True)
    company_df.ta.macd(append=True)
    company_df.ta.atr(length=14, append=True)  # ✅ NEW: Added ATR(14)
    print("Indicators calculated successfully.")

    # --- Part 4: Generate the Prompt ---
    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    filtered_df = company_df[company_df.index <= analysis_date]

    if filtered_df.empty:
        return f"Error: No historical data found for {company_ticker} on or before {analysis_date_str}."

    latest_data = filtered_df.iloc[-1]

    current_price = round(latest_data['Close'], 2)
    sma_50 = round(latest_data.get('SMA_50', 0), 2)
    sma_200 = round(latest_data.get('SMA_200', 0), 2)
    rsi_14 = round(latest_data.get('RSI_14', 0), 2)
    macd_line = latest_data.get('MACD_12_26_9', 0)
    signal_line = latest_data.get('MACDs_12_26_9', 0)
    atr_14 = round(latest_data.get('ATRr_14', latest_data.get('ATR_14', 0)), 2)  # ✅ NEW: Extract ATR(14)

    if pd.isna(macd_line) or pd.isna(signal_line):
        macd_signal = "Not Available"
    elif macd_line > signal_line:
        macd_signal = "Bullish Crossover"
    else:
        macd_signal = "Bearish Crossover"

    final_prompt = f"""
**Persona:**
You are a 'Technical Analyst Agent'. Your task is to analyze a stock's price action and key technical indicators to determine its current trend and momentum. Ignore all fundamental data, news, and macroeconomic conditions.

---

**Context: Technical Data for {company_ticker} on {analysis_date_str}**

**1. Price Context:**
* **Current Price:** {current_price}

**2. Key Technical Indicators:**
* **50-Day Simple Moving Average (SMA):** {sma_50}
* **200-Day Simple Moving Average (SMA):** {sma_200}
* **Relative Strength Index (RSI - 14 day):** {rsi_14}
* **MACD Signal:** {macd_signal}
* **ATR(14):** {atr_14}  # ✅ NEW: Added ATR(14) to prompt

---

**Task:**
Based strictly on the technical data provided above, generate a structured analysis. Output your response in the following JSON format:

{{
  "technical_view": "Provide the overall technical view (e.g., 'Bullish Trend', 'Bearish Momentum', 'Overbought', 'Oversold', 'Neutral').",
  "confidence_score": "Provide a confidence score from 1 (low) to 10 (high).",
  "rationale": "Provide a concise, one-paragraph summary explaining how the indicators support your conclusion. For example, mention if the price is above key moving averages or if the RSI indicates an overbought condition."
}}
"""
    return final_prompt

# --- How to Use the New Function ---
if __name__ == "__main__":
    # Ensure this points to your new, clean CSV file
    stock_data_file = 'stock_prices.csv'
    
    ticker_to_analyze = 'ASML NA Equity'
    date_to_analyze = "2010-05-20"
    
    llm_prompt = generate_technical_prompt(date_to_analyze, stock_data_file, ticker_to_analyze)
    
    print("\n--- Generated Prompt for LLM API ---")
    print(llm_prompt)