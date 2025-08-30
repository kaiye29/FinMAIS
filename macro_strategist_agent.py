import pandas as pd
from datetime import datetime
import google.generativeai as genai
import json
import os

# --- Paste your Google AI Studio API Key here ---
API_KEY = 'AIzaSyDSyen4rBDBcUuG-uZb2G6B3CPjw6k_wUQ'
genai.configure(api_key=API_KEY)

# Global cache for macro recommendations
MACRO_CACHE = {}

def get_interest_rate_analysis(analysis_date_str: str, data_filepath: str) -> str:
    """Analyzes interest rate data and returns a summary string."""
    try:
        df = pd.read_csv(data_filepath)
        df.columns = ['DATE', 'TIME_PERIOD', 'RATE']
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')
        df.dropna(subset=['TIME_PERIOD'], inplace=True)
    except FileNotFoundError:
        return "Error: Interest rate data file not found."
    
    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    relevant_data = df[df['TIME_PERIOD'] <= analysis_date].sort_values(by='TIME_PERIOD', ascending=False)
    if relevant_data.empty: return f"No interest rate data found on or before {analysis_date_str}."
    
    current_rate = relevant_data.iloc[0]['RATE']
    if len(relevant_data) < 2: return f"The ECB interest rate was {current_rate}%. Not enough prior data for trend."
    
    previous_rate = relevant_data.iloc[1]['RATE']
    if current_rate > previous_rate: 
        trend_text = f"This reflects a recent hike of {round(current_rate - previous_rate, 2)}%, signaling a hawkish policy stance."
    elif current_rate < previous_rate: 
        trend_text = f"This reflects a recent cut of {round(previous_rate - current_rate, 2)}%, signaling a dovish policy stance."
    else: 
        trend_text = "The rate was held unchanged from the previous decision, suggesting a neutral or 'wait-and-see' approach."
    
    return f"The key ECB interest rate is {current_rate}%. {trend_text}"

def get_inflation_analysis(analysis_date_str: str, data_filepath: str) -> str:
    """Analyzes inflation data and returns a summary string."""
    try:
        df = pd.read_csv(data_filepath)
        df['Date'] = pd.to_datetime(df['Date'].str.replace('MM', '-'), format='%Y-%m').dt.to_period('M').dt.end_time
    except FileNotFoundError:
        return "Error: Inflation data file not found."
    
    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    relevant_data = df[df['Date'] <= analysis_date].sort_values(by='Date', ascending=False)
    if relevant_data.empty: return f"No inflation data found on or before {analysis_date_str}."
    
    current_rate = relevant_data.iloc[0]['CPI_Annual_Rate']
    if len(relevant_data) < 2: return f"The inflation rate was {current_rate}%. Not enough prior data for trend."
    
    previous_rate = relevant_data.iloc[1]['CPI_Annual_Rate']
    if current_rate > 2.0: target_comparison = "which is above the ECB's 2% target"
    elif current_rate < 2.0: target_comparison = "which is below the ECB's 2% target"
    else: target_comparison = "which is at the ECB's 2% target"
    
    if current_rate > previous_rate: 
        momentum_text = f"This shows an acceleration from the previous month's {previous_rate}%."
    elif current_rate < previous_rate: 
        momentum_text = f"This shows a deceleration from the previous month's {previous_rate}%."
    else: 
        momentum_text = "The rate is stable compared to the previous month."
    
    return f"The annual CPI is {current_rate}%, {target_comparison}. {momentum_text}"

def get_sector_industry_structure(company_filepath: str) -> dict:
    """Reads the company file and returns a dictionary of {sector: [industries]}."""
    try:
        df = pd.read_csv(company_filepath)
        structure = df.groupby('Sector')['Industry'].unique().apply(list).to_dict()
        return structure
    except FileNotFoundError:
        return {"Error": "Company sector/industry file not found."}

def call_gemini_api_for_macro(prompt: str):
    """Calls Gemini API and returns parsed JSON response."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        response = model.generate_content(prompt)
        json_response = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(json_response)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def get_cached_macro_recommendations(analysis_date: str, interest_rate_file: str, inflation_file: str, company_file: str):
    """
    Gets macro recommendations for a specific date. Uses cache to ensure consistency.
    """
    cache_key = analysis_date
    
    # Check if we already have recommendations for this date
    if cache_key in MACRO_CACHE:
        print(f"Using cached macro recommendations for {analysis_date}")
        return MACRO_CACHE[cache_key]
    
    print(f"Generating new macro recommendations for {analysis_date}")
    
    # Generate the prompt
    interest_summary = get_interest_rate_analysis(analysis_date, interest_rate_file)
    inflation_summary = get_inflation_analysis(analysis_date, inflation_file)
    sector_structure = get_sector_industry_structure(company_file)
    
    structure_str = json.dumps(sector_structure, indent=2)
    
    prompt = f"""
**Persona:**
You are a 'Macro Strategist Agent' for a multi-agent investment fund. Your role is to analyze top-down macroeconomic data to determine the overall market regime and provide nuanced, industry-level strategic recommendations.

---

**Context: Macroeconomic Data for {analysis_date}**

* **Central Bank Posture:** {interest_summary}
* **Inflation Report:** {inflation_summary}

---

**Strategic Framework & Task:**

Your analysis must be guided by the general principles of macroeconomics. For the data provided, determine the overall market regime (e.g., 'Stagflationary Risk', 'Healthy Growth', etc.).

Your main task is to provide a **nuanced, industry-level recommendation**. Do not just give a single recommendation for an entire sector. Instead, for each sector, you must analyze the specific industries within it and provide a differentiated recommendation (e.g., Strong Overweight, Overweight, Neutral, Underweight, Strong Underweight).

**You must provide a specific, one-sentence reason for each industry's recommendation**, explaining why it might perform differently from its peers. For example, within Technology, you might argue that 'Semiconductor Equipment' is more cyclical and thus a 'Strong Underweight', while 'Software - Infrastructure' is more resilient due to recurring revenue models and thus only a 'Slight Underweight'.

**CRITICAL: Each industry should receive EXACTLY ONE recommendation that will apply to ALL companies in that industry on this date.**

---

**Investment Universe (Sectors and Industries):**
```json
{structure_str}
```

---

**Task:**
Based on the data, framework, and investment universe provided, generate a structured analysis. Output your response in the following JSON format:

{{
  "market_regime": "Provide the overall market regime (e.g., 'Stagflationary Risk', 'Restrictive Neutral').",
  "overall_rationale": "Provide a concise, one-paragraph summary explaining your reasoning for the market regime and your high-level strategic approach.",
  "industry_recommendations": [
    {{
      "sector": "Sector Name",
      "industry": "Industry Name",
      "recommendation": "Your differentiated recommendation (e.g., 'Strong Underweight')",
      "rationale": "Your specific, one-sentence reason for this industry's rating."
    }}
  ]
}}
"""
    
    # Call the API
    response = call_gemini_api_for_macro(prompt)
    
    if response:
        # Cache the response
        MACRO_CACHE[cache_key] = response
        # Optional: Save cache to file for persistence
        save_cache_to_file()
        return response
    else:
        return None

def save_cache_to_file():
    """Save macro cache to file for persistence across runs."""
    try:
        with open('macro_cache.json', 'w') as f:
            json.dump(MACRO_CACHE, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save cache to file: {e}")

def load_cache_from_file():
    """Load macro cache from file if it exists."""
    global MACRO_CACHE
    try:
        if os.path.exists('macro_cache.json'):
            with open('macro_cache.json', 'r') as f:
                MACRO_CACHE = json.load(f)
            print(f"Loaded {len(MACRO_CACHE)} cached macro recommendations")
    except Exception as e:
        print(f"Warning: Could not load cache from file: {e}")
        MACRO_CACHE = {}

def get_industry_recommendation_for_ticker(analysis_date: str, ticker: str, company_file: str):
    """
    Gets the macro recommendation for a specific ticker's industry.
    """
    try:
        # Load company data to find the ticker's industry
        df = pd.read_csv(company_file)
        company_info = df[df['Ticker'] == ticker]
        
        if company_info.empty:
            return None, f"Ticker {ticker} not found in company database"
        
        industry = company_info.iloc[0]['Industry']
        sector = company_info.iloc[0]['Sector']
        
        # Get cached macro recommendations
        macro_data = get_cached_macro_recommendations(
            analysis_date, 'ECB_Data.csv', 'CPI_Raw.csv', company_file
        )
        
        if not macro_data:
            return None, "Could not get macro recommendations"
        
        # Find the recommendation for this industry
        for rec in macro_data.get('industry_recommendations', []):
            if rec['industry'] == industry:
                return {
                    'sector': sector,
                    'industry': industry,
                    'recommendation': rec['recommendation'],
                    'rationale': rec['rationale'],
                    'market_regime': macro_data['market_regime']
                }, None
        
        return None, f"No macro recommendation found for industry: {industry}"
        
    except Exception as e:
        return None, f"Error getting industry recommendation: {e}"

# Initialize cache on import
load_cache_from_file()

# ==============================================================================
# Legacy function for backward compatibility
# ==============================================================================
def get_macro_view_prompt(analysis_date: str, interest_rate_file: str, inflation_file: str, company_file: str) -> str:
    """
    Legacy function maintained for backward compatibility.
    Now returns a simplified prompt since we handle macro recommendations separately.
    """
    interest_summary = get_interest_rate_analysis(analysis_date, interest_rate_file)
    inflation_summary = get_inflation_analysis(analysis_date, inflation_file)
    
    return f"""
**Context: Macroeconomic Data for {analysis_date}**

* **Central Bank Posture:** {interest_summary}
* **Inflation Report:** {inflation_summary}

**Note:** Industry-specific macro recommendations are now handled separately for consistency.
"""

if __name__ == "__main__":
    # Test the new functionality
    test_date = "2023-01-15"
    test_ticker = "ASML NA Equity"
    
    recommendation, error = get_industry_recommendation_for_ticker(
        test_date, test_ticker, 'Company_Sector_Industry.csv'
    )
    
    if recommendation:
        print("Macro recommendation for", test_ticker, ":")
        print(json.dumps(recommendation, indent=2))
    else:
        print("Error:", error)