import pandas as pd
from datetime import datetime
import random
import numpy as np

# Import the prompt-generating functions from your other agent files
from macro_strategist_agent import get_industry_recommendation_for_ticker
from fundamental_analysis_agent_improved import generate_fundamental_prompt_simple 
from technical_analyst_agent import generate_technical_prompt

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def generate_final_decision_prompt_IMPROVED(
    analysis_date_str: str, 
    company_ticker: str, 
    use_ratios_data=None, 
    use_prices_data=None,
    random_seed=42  # NEW: Add random seed parameter
) -> dict:
    """
    Generates a final decision prompt using the IMPROVED fundamental agent with SHAP,
    and the new consistent macro strategist agent with random seed control.
    Returns it along with the top feature identified by SHAP for testing.
    """
    print(f"--- Starting IMPROVED Multi-Agent Analysis (Seed: {random_seed}) ---")
    
    # Set random seeds at the start
    set_random_seeds(random_seed)
    
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
        # Set default values for logging
        macro_recommendation = {
            'recommendation': 'Neutral',
            'rationale': f'Error: {macro_error}'
        }

    print(f"\n2. Running IMPROVED Fundamental Analyst Agent (with SHAP, seed: {random_seed})...")
    
    # Handle the dictionary output from the improved fundamental agent with seed control
    fundamental_response = generate_fundamental_prompt_simple(
        analysis_date_str, 
        ratios_data_file, 
        price_data_file, 
        company_ticker, 
        use_ratios_data=use_ratios_data, 
        use_prices_data=use_prices_data,
        random_seed=random_seed  # NEW: Pass the seed
    )
    
    if "Error" in fundamental_response:
        fundamental_context = f"Context: Fundamental Data\n\nAnalysis failed: {fundamental_response['Error']}"
        top_feature_for_perturbation = None
        training_samples = 0
        features_used = 0
    else:
        fundamental_prompt = fundamental_response['prompt']
        top_feature_for_perturbation = fundamental_response['top_feature']
        fundamental_context = fundamental_prompt.split("---")[1].strip() if "---" in fundamental_prompt else fundamental_prompt
        training_samples = fundamental_response.get('training_samples', 0)
        features_used = fundamental_response.get('features_used', 0)

    print("\n3. Running Technical Analyst Agent...")
    technical_prompt = generate_technical_prompt(analysis_date_str, price_data_file, company_ticker)
    if "Error" in technical_prompt:
        technical_context = f"Context: Technical Data\n\nAnalysis failed: {technical_prompt}"
    else:
        technical_context = technical_prompt.split("---")[1].strip() if "---" in technical_prompt else technical_prompt
    
    print(f"\n--- All Agent Analyses Complete (Seed: {random_seed}) ---")

    final_decision_prompt = f"""
**Persona:**
You are the 'Decision Engine Agent', acting as a Chief Investment Officer. Your task is to synthesize the analyses from three specialist agents to make a final, decisive trading recommendation. You must weigh any conflicting opinions and provide a clear, actionable decision.

**Random Seed:** {random_seed} (for reproducibility)

---

**Investment Committee Meeting Brief: Analysis for {company_ticker} on {analysis_date_str}**

**1. Macro Strategist's Industry-Level View:**
{macro_context}

**2. Fundamental Analyst's Company-Level View (with SHAP, Seed: {random_seed}):**
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
  "final_rationale": "Provide a concise, one-paragraph summary explaining how you weighted the macro industry view, company fundamentals (including SHAP insights), and technical signals in your decision.",
  "macro_influence": "Briefly explain how the macro industry recommendation influenced your decision (e.g., 'Macro overweight supported the buy decision' or 'Ignored macro underweight due to strong fundamentals')."
}}
"""
    
    # Return a dictionary with the prompt, top feature, AND macro info for logging
    return {
        "prompt": final_decision_prompt,
        "top_feature": top_feature_for_perturbation,
        "macro_recommendation": macro_recommendation['recommendation'],
        "macro_rationale": macro_recommendation['rationale'],
        "random_seed": random_seed,  # NEW: Include seed in metadata
        "xgboost_training_samples": training_samples,  # NEW
        "xgboost_features_used": features_used  # NEW
    }


def generate_final_decision_prompt_IMPROVED_with_temporal_awareness(
    analysis_date_str: str, 
    company_ticker: str, 
    use_ratios_data=None, 
    use_prices_data=None,
    temporal_protection_enabled=True,
    random_seed=42  # NEW: Add random seed parameter
) -> dict:
    """
    Enhanced: Same as your original function but with temporal protection awareness and seed control.
    """
    print(f"--- Starting IMPROVED Multi-Agent Analysis {'(Temporal Protected)' if temporal_protection_enabled else ''} (Seed: {random_seed}) ---")
    
    # Set random seeds at the start
    set_random_seeds(random_seed)
    
    # Minimal addition: Validate temporal protection if enabled
    if temporal_protection_enabled and use_ratios_data is not None:
        analysis_date = pd.to_datetime(analysis_date_str)
        max_ratios_date = use_ratios_data['Date'].max()
        max_prices_date = use_prices_data['Date'].max() if use_prices_data is not None else analysis_date
        
        if max_ratios_date > analysis_date or max_prices_date > analysis_date:
            print(f"Warning: Potential temporal leakage detected!")
            print(f"   Analysis date: {analysis_date_str}")
            print(f"   Max ratios date: {max_ratios_date}")
            print(f"   Max prices date: {max_prices_date}")
        else:
            print(f"Temporal protection validated - no future data detected")
    
    # REST OF YOUR EXISTING CODE STAYS THE SAME
    ratios_data_file = 'financial_ratios_clean_updated.csv'
    price_data_file = 'stock_prices.csv'
    company_list_file = 'Company_Sector_Industry.csv'
    
    # --- Step 1: Get Macro Analysis (unchanged) ---
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
        macro_recommendation = {
            'recommendation': 'Neutral',
            'rationale': f'Error: {macro_error}'
        }

    print(f"\n2. Running IMPROVED Fundamental Analyst Agent (with SHAP{'+ Temporal Protection' if temporal_protection_enabled else ''}, seed: {random_seed})...")
    
    # Handle the dictionary output from the improved fundamental agent with seed control
    fundamental_response = generate_fundamental_prompt_simple(
        analysis_date_str, 
        ratios_data_file, 
        price_data_file, 
        company_ticker, 
        use_ratios_data=use_ratios_data, 
        use_prices_data=use_prices_data,
        random_seed=random_seed  # NEW: Pass the seed
    )
    
    if "Error" in fundamental_response:
        fundamental_context = f"Context: Fundamental Data\n\nAnalysis failed: {fundamental_response['Error']}"
        top_feature_for_perturbation = None
        training_samples = 0
        features_used = 0
        temporal_status = "error"
    else:
        fundamental_prompt = fundamental_response['prompt']
        top_feature_for_perturbation = fundamental_response['top_feature']
        fundamental_context = fundamental_prompt.split("---")[1].strip() if "---" in fundamental_prompt else fundamental_prompt
        training_samples = fundamental_response.get('training_samples', 0)
        features_used = fundamental_response.get('features_used', 0)
        temporal_status = fundamental_response.get('temporal_protection', 'unknown')

    print("\n3. Running Technical Analyst Agent...")
    # Minimal addition: Handle temporal protection for technical analysis
    if temporal_protection_enabled and use_prices_data is not None:
        # Save filtered data temporarily
        use_prices_data.to_csv('temp_filtered_prices.csv', index=False)
        technical_prompt = generate_technical_prompt(analysis_date_str, 'temp_filtered_prices.csv', company_ticker)
    else:
        technical_prompt = generate_technical_prompt(analysis_date_str, price_data_file, company_ticker)
        
    if "Error" in technical_prompt:
        technical_context = f"Context: Technical Data\n\nAnalysis failed: {technical_prompt}"
    else:
        technical_context = technical_prompt.split("---")[1].strip() if "---" in technical_prompt else technical_prompt
    
    print(f"\n--- All Agent Analyses Complete {'with Temporal Protection' if temporal_protection_enabled else ''} (Seed: {random_seed}) ---")

    # Minimal addition: Enhanced prompt with temporal status and seed
    temporal_header = f"\n**Temporal Protection: {'Enabled' if temporal_protection_enabled else 'Standard Mode'}**" if temporal_protection_enabled else ""
    seed_header = f"\n**Random Seed: {random_seed}** (for reproducibility)"

    final_decision_prompt = f"""
**Persona:**
You are the 'Decision Engine Agent', acting as a Chief Investment Officer. Your task is to synthesize the analyses from three specialist agents to make a final, decisive trading recommendation. You must weigh any conflicting opinions and provide a clear, actionable decision.{temporal_header}{seed_header}

---

**Investment Committee Meeting Brief: Analysis for {company_ticker} on {analysis_date_str}**

**1. Macro Strategist's Industry-Level View:**
{macro_context}

**2. Fundamental Analyst's Company-Level View (with SHAP, Seed: {random_seed}):**
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
  "final_rationale": "Provide a concise, one-paragraph summary explaining how you weighted the macro industry view, company fundamentals (including SHAP insights), and technical signals in your decision.",
  "macro_influence": "Briefly explain how the macro industry recommendation influenced your decision (e.g., 'Macro overweight supported the buy decision' or 'Ignored macro underweight due to strong fundamentals')."
}}
"""
    
    # Enhanced return with temporal metadata and seed info
    return {
        "prompt": final_decision_prompt,
        "top_feature": top_feature_for_perturbation,
        "macro_recommendation": macro_recommendation['recommendation'],
        "macro_rationale": macro_recommendation['rationale'],
        "temporal_protection": "enabled" if temporal_protection_enabled else "standard",
        "xgboost_training_samples": training_samples,
        "xgboost_features_used": features_used,
        "temporal_status": temporal_status,
        "random_seed": random_seed  # NEW: Include seed in metadata
    }


# --- Example of how to run this file directly ---
if __name__ == "__main__":
    
    ticker_to_analyze = 'ASML NA Equity'
    date_to_analyze = "2024-05-20"
    
    # Test both functions with multiple seeds (updated to 5 seeds)
    test_seeds = [42, 123, 456, 789, 999]
    
    for seed in test_seeds:
        print(f"\n{'='*60}")
        print(f"Testing with seed {seed}")
        print(f"{'='*60}")
        
        print("Testing Standard Mode:")
        response_dict = generate_final_decision_prompt_IMPROVED(
            date_to_analyze, ticker_to_analyze, random_seed=seed
        )
        
        if response_dict.get("top_feature"):
            print("\n--- Standard Mode Results ---")
            print(f"Top Feature: {response_dict['top_feature']}")
            print(f"Macro Recommendation: {response_dict['macro_recommendation']}")
            print(f"Seed Used: {response_dict['random_seed']}")
            print(f"XGBoost Training Samples: {response_dict.get('xgboost_training_samples', 'N/A')}")
        else:
            print("\n--- An error occurred during standard analysis ---")
        
        print("\nTesting Temporal Protection Mode:")
        try:
            # Load sample data for temporal protection test
            price_data = pd.read_csv('stock_prices.csv', parse_dates=['Date'])
            ratios_data = pd.read_csv('financial_ratios_clean_updated.csv', parse_dates=['Date'])
            
            # Filter to analysis date
            analysis_date = pd.to_datetime(date_to_analyze)
            filtered_price = price_data[price_data['Date'] <= analysis_date]
            filtered_ratios = ratios_data[ratios_data['Date'] <= analysis_date]
            
            temporal_response = generate_final_decision_prompt_IMPROVED_with_temporal_awareness(
                date_to_analyze, 
                ticker_to_analyze,
                use_ratios_data=filtered_ratios,
                use_prices_data=filtered_price,
                temporal_protection_enabled=True,
                random_seed=seed  # NEW: Pass seed
            )
            
            print(f"Temporal Mode - Top Feature: {temporal_response['top_feature']}")
            print(f"Temporal Mode - Protection Status: {temporal_response['temporal_protection']}")
            print(f"Temporal Mode - XGBoost Samples: {temporal_response['xgboost_training_samples']}")
            print(f"Temporal Mode - Seed Used: {temporal_response['random_seed']}")
            
        except FileNotFoundError as e:
            print(f"Could not test temporal protection mode: {e}")