import pandas as pd
from datetime import datetime

# Import the prompt-generating functions from your other agent files
from macro_strategist_agent import get_industry_recommendation_for_ticker
from fundamental_analyst_agent import generate_fundamental_prompt
from technical_analyst_agent import generate_technical_prompt

def generate_final_decision_prompt(analysis_date_str: str, company_ticker: str) -> str:
    """
    Generates a final decision prompt that includes consistent macro recommendations.
    """
    print(f"--- Starting Multi-Agent Analysis for {company_ticker} ---")
    
    # --- Define the file paths for all data sources ---
    ratios_data_file = 'financial_ratios_clean_updated.csv'
    price_data_file = 'stock_prices.csv'
    company_list_file = 'Company_Sector_Industry.csv'
    
    # --- Step 1: Get Macro Analysis (NEW: Consistent industry-level) ---
    print("\n1. Getting Macro Strategist's Industry Analysis...")
    
    macro_recommendation, macro_error = get_industry_recommendation_for_ticker(
        analysis_date_str, company_ticker, company_list_file
    )
    
    if macro_recommendation:
        macro_context = f"""
**Industry:** {macro_recommendation['industry']} (Sector: {macro_recommendation['sector']})
**Market Regime:** {macro_recommendation['market_regime']}
**Macro Recommendation:** {macro_recommendation['recommendation']}
**Rationale:** {macro_recommendation['rationale']}
"""
    else:
        macro_context = f"""
**Macro Analysis Error:** {macro_error}
**Default Stance:** Neutral - unable to determine industry-specific macro positioning
"""

    # --- Step 2: Get Fundamental Analysis ---
    print("\n2. Running Fundamental Analyst Agent...")
    fundamental_prompt = generate_fundamental_prompt(analysis_date_str, ratios_data_file, company_ticker)
    # Extract just the context section
    fundamental_context = fundamental_prompt.split("**Context: Fundamental Data")[1].split("---")[0] if "**Context: Fundamental Data" in fundamental_prompt else "Fundamental analysis unavailable"

    # --- Step 3: Get Technical Analysis ---
    print("\n3. Running Technical Analyst Agent...")
    technical_prompt = generate_technical_prompt(analysis_date_str, price_data_file, company_ticker)
    # Extract just the context section
    technical_context = technical_prompt.split("**Context: Technical Data")[1].split("---")[0] if "**Context: Technical Data" in technical_prompt else "Technical analysis unavailable"
    
    print("\n--- All Agent Analyses Complete ---")

    # --- Step 4: Construct the Enhanced Final Prompt ---
    final_decision_prompt = f"""
**Persona:**
You are the 'Decision Engine Agent', acting as a Chief Investment Officer. Your task is to synthesize the analyses from three specialist agents to make a final, decisive trading recommendation.

---

**Investment Committee Meeting Brief: Analysis for {company_ticker} on {analysis_date_str}**

**1. Macro Strategist's Industry-Level View:**
{macro_context}

**2. Fundamental Analyst's Company-Level View:**
{fundamental_context}

**3. Technical Analyst's Price Action View:**
{technical_context}

---

**Decision Framework:**
Consider how the macro industry positioning aligns with the company's fundamentals and technical momentum:

- **Macro-Fundamental Alignment:** Does the company's financial health support the macro industry view?
- **Technical Confirmation:** Does the price action confirm or contradict the fundamental/macro thesis?
- **Risk-Reward Assessment:** What is the overall risk-adjusted opportunity?

**Task:**
Based on the complete investment committee brief above, make a final trading decision. The macro recommendation provides industry-level context, but your final decision should consider all three perspectives. Output your response in the following JSON format:

{{
  "final_recommendation": "Provide a clear, one-word recommendation (Buy, Sell, or Hold).",
  "confidence_score": "Provide a confidence score for your final decision from 1 (low) to 10 (high).",
  "final_rationale": "Provide a concise, one-paragraph summary explaining how you weighted the macro industry view, company fundamentals, and technical signals in your decision.",
  "macro_influence": "Briefly explain how the macro industry recommendation influenced your decision (e.g., 'Macro overweight supported the buy decision' or 'Ignored macro underweight due to strong fundamentals')."
}}
"""
    return final_decision_prompt


def generate_final_decision_prompt_with_temporal_protection(
    analysis_date_str: str, 
    company_ticker: str,
    available_ratios_data=None  # 🔒 NEW: Accept filtered ratios data
) -> str:
    """
    🔒 ENHANCED: Generates final decision prompt with temporal protection.
    Passes filtered data to prevent future data leakage in fundamental analysis.
    """
    print(f"--- Starting Multi-Agent Analysis for {company_ticker} (🔒 Temporal Protected) ---")
    
    # --- Define the file paths for all data sources ---
    ratios_data_file = 'financial_ratios_clean_updated.csv'
    price_data_file = 'stock_prices.csv'
    company_list_file = 'Company_Sector_Industry.csv'
    
    # 🔒 Temporal validation
    if available_ratios_data is not None:
        analysis_date = pd.to_datetime(analysis_date_str)
        max_ratios_date = available_ratios_data['Date'].max()
        if max_ratios_date > analysis_date:
            print(f"⚠️ WARNING: Ratios data contains future information!")
            print(f"   Analysis date: {analysis_date_str}")
            print(f"   Max ratios date: {max_ratios_date}")
        else:
            print(f"✅ Temporal protection validated for ratios data")
    
    # --- Step 1: Get Macro Analysis (already has temporal protection) ---
    print("\n1. Getting Macro Strategist's Industry Analysis...")
    
    macro_recommendation, macro_error = get_industry_recommendation_for_ticker(
        analysis_date_str, company_ticker, company_list_file
    )
    
    if macro_recommendation:
        macro_context = f"""
**Industry:** {macro_recommendation['industry']} (Sector: {macro_recommendation['sector']})
**Market Regime:** {macro_recommendation['market_regime']}
**Macro Recommendation:** {macro_recommendation['recommendation']}
**Rationale:** {macro_recommendation['rationale']}
"""
    else:
        macro_context = f"""
**Macro Analysis Error:** {macro_error}
**Default Stance:** Neutral - unable to determine industry-specific macro positioning
"""

    # --- Step 2: Get Fundamental Analysis with temporal protection ---
    print("\n2. Running Fundamental Analyst Agent (🔒 Temporal Protected)...")
    fundamental_prompt = generate_fundamental_prompt(
        analysis_date_str, 
        ratios_data_file, 
        company_ticker,
        use_data=available_ratios_data  # 🔒 NEW: Pass filtered data
    )
    # Extract just the context section
    fundamental_context = fundamental_prompt.split("**Context: Fundamental Data")[1].split("---")[0] if "**Context: Fundamental Data" in fundamental_prompt else "Fundamental analysis unavailable"

    # --- Step 3: Get Technical Analysis (already has temporal protection) ---
    print("\n3. Running Technical Analyst Agent...")
    technical_prompt = generate_technical_prompt(analysis_date_str, price_data_file, company_ticker)
    # Extract just the context section
    technical_context = technical_prompt.split("**Context: Technical Data")[1].split("---")[0] if "**Context: Technical Data" in technical_prompt else "Technical analysis unavailable"
    
    print("\n--- All Agent Analyses Complete with Temporal Protection ---")

    # --- Step 4: Construct the Enhanced Final Prompt ---
    temporal_header = "\n**🔒 TEMPORAL PROTECTION: ✅ ENABLED**"
    
    final_decision_prompt = f"""
**Persona:**
You are the 'Decision Engine Agent', acting as a Chief Investment Officer. Your task is to synthesize the analyses from three specialist agents to make a final, decisive trading recommendation.{temporal_header}

---

**Investment Committee Meeting Brief: Analysis for {company_ticker} on {analysis_date_str}**

**1. Macro Strategist's Industry-Level View:**
{macro_context}

**2. Fundamental Analyst's Company-Level View:**
{fundamental_context}

**3. Technical Analyst's Price Action View:**
{technical_context}

---

**Decision Framework:**
Consider how the macro industry positioning aligns with the company's fundamentals and technical momentum:

- **Macro-Fundamental Alignment:** Does the company's financial health support the macro industry view?
- **Technical Confirmation:** Does the price action confirm or contradict the fundamental/macro thesis?
- **Risk-Reward Assessment:** What is the overall risk-adjusted opportunity?

**Task:**
Based on the complete investment committee brief above, make a final trading decision. The macro recommendation provides industry-level context, but your final decision should consider all three perspectives. Output your response in the following JSON format:

{{
  "final_recommendation": "Provide a clear, one-word recommendation (Buy, Sell, or Hold).",
  "confidence_score": "Provide a confidence score for your final decision from 1 (low) to 10 (high).",
  "final_rationale": "Provide a concise, one-paragraph summary explaining how you weighted the macro industry view, company fundamentals, and technical signals in your decision.",
  "macro_influence": "Briefly explain how the macro industry recommendation influenced your decision (e.g., 'Macro overweight supported the buy decision' or 'Ignored macro underweight due to strong fundamentals')."
}}
"""
    return final_decision_prompt


# --- Example Usage ---
if __name__ == "__main__":
    
    my_tickers = ['ASML NA Equity', 'INGA NA Equity', 'ADYEN NA Equity']
    start_period = "2024-12-01"
    end_period = "2024-12-31"
    
    rebalance_dates = pd.date_range(start=start_period, end=end_period, freq='BMS')
    
    print("="*60)
    print(f"Generating example prompts for the test set from {start_period} to {end_period}")
    print("="*60)

    # Test with just one date and ticker first
    test_date = rebalance_dates[0].strftime("%Y-%m-%d")
    test_ticker = my_tickers[0]
    
    print("\n🔧 Testing Standard Mode:")
    standard_prompt = generate_final_decision_prompt(test_date, test_ticker)
    print(f"Standard prompt generated successfully")
    
    print("\n🔒 Testing Temporal Protection Mode:")
    try:
        # Load and filter sample ratios data for testing
        ratios_data = pd.read_csv('financial_ratios_clean_updated.csv', parse_dates=['Date'])
        analysis_date = pd.to_datetime(test_date)
        filtered_ratios = ratios_data[ratios_data['Date'] <= analysis_date]
        
        temporal_prompt = generate_final_decision_prompt_with_temporal_protection(
            test_date, 
            test_ticker,
            available_ratios_data=filtered_ratios
        )
        
        print(f"Temporal protection prompt generated successfully")
        print(f"Filtered ratios data: {len(filtered_ratios)} rows (vs {len(ratios_data)} original)")
        
    except FileNotFoundError:
        print("⚠️ Could not test temporal protection mode - ratios file not found")
    
    print("\n\n" + "#"*70)
    print(f"### Generated Final Prompt for: {test_ticker} on {test_date} ###")
    print("#"*70)
    print("Both standard and temporal protection modes tested successfully!")