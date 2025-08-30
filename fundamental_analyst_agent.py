import pandas as pd
from datetime import datetime

def get_ratios_for_ticker(df: pd.DataFrame, ticker: str, analysis_date: datetime) -> dict:
    """
    Extracts the latest available financial ratios for a specific ticker on or before a given date.
    """
    # Filter for the specific ticker and for dates on or before the analysis date
    ticker_data = df[(df['Ticker'] == ticker) & (df['Date'] <= analysis_date)]
    
    if ticker_data.empty:
        return {}
        
    # Get the most recent data point available
    latest_data = ticker_data.sort_values(by='Date', ascending=False).iloc[0]
    
    # Get all ratios for that date and ticker
    ratios_on_date = df[(df['Ticker'] == ticker) & (df['Date'] == latest_data['Date'])]
    
    # Create a dictionary of the ratios
    ratios_dict = pd.Series(ratios_on_date['Value'].values, index=ratios_on_date['Field']).to_dict()
    return ratios_dict


def generate_fundamental_prompt(analysis_date_str: str, data_filepath: str, company_ticker: str, use_data=None) -> str:
    """
    Analyzes a company's financial ratios and generates a prompt for the Gemini API.
    """
    try:
        # Use the provided DataFrame for testing, or load from file
        df = use_data if use_data is not None else pd.read_csv(data_filepath)
        df = df.copy()  # Make explicit copy first
        df['Date'] = pd.to_datetime(df['Date'])
    except FileNotFoundError:
        return "Error: Financial ratios data file not found."

    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    
    # Get the latest ratios for the specified ticker
    ratios = get_ratios_for_ticker(df, company_ticker, analysis_date)

    if not ratios:
        return f"Error: No fundamental data found for {company_ticker} on or before {analysis_date_str}."

    # --- FIX: Use the exact field names from the CSV file ---

    # Valuation Ratios
    pe_ratio = ratios.get('Price Earnings Ratio (P/E)', 'N/A')
    pb_ratio = ratios.get('Price to Book Ratio', 'N/A')
    dividend_yield = ratios.get('Dividend 12 Month Yield', 'N/A')

    # Profitability & Growth Ratios
    profit_margin = ratios.get('Profit Margin', 'N/A')
    roe = ratios.get('Return on Common Equity', 'N/A')
    sales_growth = ratios.get('Revenue Growth Year over Year', 'N/A')

    # Liquidity & Solvency Ratios
    current_ratio = ratios.get('Current Ratio', 'N/A')
    debt_to_equity = ratios.get('Total Debt to Total Equity', 'N/A')
    fcf_to_debt = ratios.get('Free Cash Flow to Total Debt', 'N/A')

    # Efficiency Ratio
    inventory_turnover = ratios.get('Inventory Turnover', 'N/A')


    # Format all 10 ratios for the prompt
    def format_ratio(value, is_percent=False):
        if isinstance(value, (int, float)):
            return f"{value:.2f}{'%' if is_percent else ''}"
        return "N/A"

    pe_ratio_str = format_ratio(pe_ratio)
    pb_ratio_str = format_ratio(pb_ratio)
    dividend_yield_str = format_ratio(dividend_yield, is_percent=True)
    profit_margin_str = format_ratio(profit_margin, is_percent=True)
    roe_str = format_ratio(roe, is_percent=True)
    sales_growth_str = format_ratio(sales_growth, is_percent=True)
    current_ratio_str = format_ratio(current_ratio)
    debt_to_equity_str = format_ratio(debt_to_equity)
    fcf_to_debt_str = format_ratio(fcf_to_debt)
    inventory_turnover_str = format_ratio(inventory_turnover)

    # The prompt now includes all 10 ratios, categorized for clarity
    final_prompt = f"""
**Persona:**
You are a 'Fundamental Analyst Agent'. Your expertise is in analyzing company-specific financial data to assess valuation, profitability, and financial health. Your analysis must be objective and based ONLY on the data provided.

---

**Context: Fundamental Data for {company_ticker} as of {analysis_date_str}**

**1. Key Financial Ratios:**

* **Valuation:**
    * P/E Ratio: {pe_ratio_str}
    * P/B Ratio: {pb_ratio_str}
    * Dividend Yield: {dividend_yield_str}

* **Profitability & Growth:**
    * Profit Margin: {profit_margin_str}
    * Return on Equity (ROE): {roe_str}
    * Sales Growth: {sales_growth_str}

* **Liquidity & Solvency:**
    * Current Ratio: {current_ratio_str}
    * Total Debt to Equity: {debt_to_equity_str}
    * FCF to Total Debt: {fcf_to_debt_str}

* **Efficiency:**
    * Inventory Turnover: {inventory_turnover_str}

---

**Task:**
Based strictly on the comprehensive financial ratios provided above, generate a structured analysis. Consider all aspects: valuation, profitability, financial health (solvency), and operational efficiency. Output your response in the following JSON format:

{{
  "fundamental_view": "Provide a one-word summary of your view (e.g., 'Bullish', 'Bearish', 'Neutral').",
  "valuation_assessment": "Assess the company's valuation (e.g., 'Appears Overvalued', 'Reasonably Priced', 'Appears Undervalued').",
  "rationale": "Provide a concise, one-sentence summary explaining your reasoning based on the full set of provided ratios."
}}
"""
    return final_prompt

# --- Example of how to run this file directly ---
if __name__ == "__main__":
    
    ticker_to_analyze = 'ASML NA Equity'
    date_to_analyze = "2024-05-20"
    # Make sure this is the correct filename for your updated ratios
    ratios_file = 'financial_ratios_clean_updated.csv'
    
    final_prompt = generate_fundamental_prompt(date_to_analyze, ratios_file, ticker_to_analyze)
    
    print("\n\n--- Generated Final Prompt for Fundamental Analyst ---")
    print(final_prompt)
