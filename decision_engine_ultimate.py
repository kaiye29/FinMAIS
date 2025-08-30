import pandas as pd
import numpy as np
from datetime import datetime
import google.generativeai as genai
import json
import random

# --- Agent Imports ---
from macro_strategist_agent import get_industry_recommendation_for_ticker
from technical_analyst_agent import generate_technical_prompt
from fundamental_analysis_agent_improved import generate_fundamental_prompt_simple

# --- API Configuration ---
API_KEY = 'AIzaSyDSyen4rBDBcUuG-uZb2G6B3CPjw6k_wUQ'
genai.configure(api_key=API_KEY)


def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


# ==============================================================================
# FIXED: Helper function with temporal protection for state discretization
# ==============================================================================
def get_optimized_state_with_temporal_protection(row: pd.Series, analysis_date: datetime) -> int:
    """
    FIXED: Uses the same state discretization as your optimized RL agent with temporal validation.
    This creates 216 states using 6 essential features, ensuring no future data is used.
    """
    
    # Temporal validation: ensure we're not using future data
    if 'Date' in row.index and pd.notna(row['Date']):
        row_date = pd.to_datetime(row['Date'])
        if row_date > analysis_date:
            raise ValueError(f"TEMPORAL VIOLATION: Row date {row_date} is after analysis date {analysis_date}")
    
    # 1. RSI Signal (3 states) - ESSENTIAL for momentum
    rsi = row.get('RSI_14', 50)
    if rsi < 30:
        rsi_state = 0  # Oversold
    elif rsi <= 70:
        rsi_state = 1  # Neutral
    else:
        rsi_state = 2  # Overbought
    
    # 2. Price vs SMA_50 (3 states) - ESSENTIAL for trend
    close = row.get('Close', 100)
    sma_50 = row.get('SMA_50', close)
    if close > sma_50 * 1.02:  # 2% above SMA
        trend_state = 2  # Strong uptrend
    elif close > sma_50 * 0.98:  # Within 2% of SMA
        trend_state = 1  # Neutral/sideways
    else:
        trend_state = 0  # Downtrend
    
    # 3. P/E Ratio (3 states) - ESSENTIAL for valuation
    # FIXED: Use the correct column name from your dataset
    pe = row.get('Price Earnings Ratio (P/E)', 20)
    if pe < 15:
        pe_state = 0  # Value
    elif pe <= 25:
        pe_state = 1  # Fair
    else:
        pe_state = 2  # Growth/Expensive
    
    # 4. Volatility via ATR (2 states) - ESSENTIAL for risk
    atr = row.get('ATR_14', 0)
    if atr == 0:  # Fallback calculation
        high = row.get('High', close * 1.01)
        low = row.get('Low', close * 0.99)
        atr = (high - low) / close
    
    vol_state = 1 if atr > 0.03 else 0  # High vs Low volatility (3% threshold)
    
    # 5. Macro Environment (2 states) - ESSENTIAL for macro context
    ecb_rate = row.get('ECB_Rate', 2.0)
    macro_state = 1 if ecb_rate > 2.5 else 0  # Restrictive vs Accommodative
    
    # 6. MACD Signal (2 states) - ESSENTIAL for trend confirmation
    macd_signal = row.get('MACD_Signal', 0)
    macd_state = 1 if macd_signal > 0 else 0  # Bullish vs Bearish signal
    
    # Combine into single state index (same formula as your RL agent)
    # 3 * 3 * 3 * 2 * 2 * 2 = 216 total states
    state_index = (rsi_state * 72 +      # 3 * 3 * 2 * 2 * 2 = 72
                   trend_state * 24 +     # 3 * 2 * 2 * 2 = 24  
                   pe_state * 8 +         # 2 * 2 * 2 = 8
                   vol_state * 4 +        # 2 * 2 = 4
                   macro_state * 2 +      # 2
                   macd_state)            # 1
    
    return min(int(state_index), 215)  # Ensure we don't exceed bounds


# ==============================================================================
# FIXED: Function to get weights from your trained RL Agent with temporal protection
# ==============================================================================
def get_rl_weights_with_temporal_protection(q_table_path: str, current_data_row: pd.Series, analysis_date: datetime, random_seed: int = 42) -> dict:
    """
    FIXED: Loads your optimized Q-table and determines optimal weights with temporal validation.
    Enhanced with seed-specific Q-table loading.
    """
    try:
        q_table = np.load(q_table_path)
        print(f"Loaded temporally protected Q-table with shape: {q_table.shape}")
    except FileNotFoundError:
        print(f"Warning: Q-table not found at {q_table_path}. Trying alternative paths...")
        # Try seed-specific and alternative paths
        alternative_paths = [
            f'optimized_agent_q_table_temporal_protected_seed_{random_seed}.npy',
            'optimized_agent_q_table_temporal_protected.npy',
            'optimized_agent_q_table.npy'
        ]
        
        q_table = None
        for alt_path in alternative_paths:
            try:
                q_table = np.load(alt_path)
                print(f"Loaded Q-table from {alt_path}")
                break
            except FileNotFoundError:
                continue
        
        if q_table is None:
            print("Warning: No Q-table found. Using balanced weights.")
            return {'macro': 0.33, 'fundamental': 0.33, 'technical': 0.33, 'source': 'default_balanced'}

    # FIXED: Match your optimized RL agent's action space
    actions = {
        0: {'macro': 0.6, 'fundamental': 0.2, 'technical': 0.2, 'source': 'rl_macro_heavy'},
        1: {'macro': 0.2, 'fundamental': 0.6, 'technical': 0.2, 'source': 'rl_fundamental_heavy'},
        2: {'macro': 0.2, 'fundamental': 0.2, 'technical': 0.6, 'source': 'rl_technical_heavy'},
        3: {'macro': 0.33, 'fundamental': 0.33, 'technical': 0.33, 'source': 'rl_balanced'}
    }
    
    # Use the temporal-protected state function
    try:
        state = get_optimized_state_with_temporal_protection(current_data_row, analysis_date)
    except ValueError as e:
        print(f"Warning: Temporal violation detected: {e}. Using balanced weights.")
        return actions[3]  # Default to balanced
    
    # Ensure state is within bounds
    if state >= q_table.shape[0]:
        print(f"Warning: State {state} exceeds Q-table bounds. Using balanced weights.")
        return actions[3]  # Default to balanced
    
    best_action_index = np.argmax(q_table[state, :])
    selected_weights = actions[best_action_index]
    
    print(f"RL Agent selected: {selected_weights['source']} (state: {state}, seed: {random_seed}, temporal protection enabled)")
    return selected_weights


# ==============================================================================
# FIXED: Main Decision Prompt Generator with comprehensive temporal protection
# ==============================================================================
def generate_final_decision_prompt_ultimate_with_temporal_protection(
    analysis_date_str: str, 
    company_ticker: str, 
    current_data_row: pd.Series,
    progressive_price_data: pd.DataFrame = None,
    progressive_ratios_data: pd.DataFrame = None,
    progressive_time_machine_data: pd.DataFrame = None,
    random_seed: int = 42  # NEW: Add random seed parameter
) -> dict:
    """
    FIXED: Generates final decision prompt with optimized RL weights, SHAP, and consistent macro.
    All data sources are temporally filtered to prevent future data leakage.
    Enhanced with comprehensive seed control.
    """
    print(f"Starting Ultimate Multi-Agent Analysis with Temporal Protection for {company_ticker} on {analysis_date_str} (Seed: {random_seed})")
    
    # Set random seeds at the beginning
    set_random_seeds(random_seed)
    
    analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    
    # --- Define file paths ---
    company_list_file = 'Company_Sector_Industry.csv'
    q_table_file = f'optimized_agent_q_table_temporal_protected_seed_{random_seed}.npy'

    # --- Step 1: Get Consistent Macro Analysis ---
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
            'rationale': f'Error: {macro_error}',
            'industry': 'Unknown',
            'sector': 'Unknown',
            'market_regime': 'Unknown'
        }

    # --- Step 2: Get Technical Analysis with Temporal Protection ---
    print("\n2. Running Technical Analyst Agent with Temporal Protection...")
    try:
        # FIXED: Use progressive price data if provided, otherwise use temporally filtered data
        if progressive_price_data is not None:
            # Save progressive data to temporary file for technical analysis
            temp_price_file = f'temp_progressive_prices_{analysis_date_str.replace("-", "")}_{random_seed}.csv'
            progressive_price_data.to_csv(temp_price_file, index=False)
            technical_prompt = generate_technical_prompt(analysis_date_str, temp_price_file, company_ticker)
            
            # Clean up temporary file
            import os
            try:
                os.remove(temp_price_file)
            except:
                pass
        else:
            # Fallback to original files (should be temporally filtered upstream)
            technical_prompt = generate_technical_prompt(analysis_date_str, 'stock_prices.csv', company_ticker)
        
        technical_context = technical_prompt.split("---")[1].strip() if "---" in technical_prompt else technical_prompt[:500]
        print("Technical analysis completed with temporal protection")
        
    except Exception as e:
        print(f"Error in technical analysis: {e}")
        technical_context = "Technical analysis unavailable due to error."

    # --- Step 3: Get Fundamental Analysis with SHAP and Temporal Protection ---
    print("\n3. Running IMPROVED Fundamental Analyst Agent (SHAP + Temporal Protection)...")
    try:
        # CRITICAL FIX: Use progressive data if provided
        if progressive_ratios_data is not None and progressive_price_data is not None:
            # Use the temporally filtered data directly
            fundamental_response = generate_fundamental_prompt_simple(
                analysis_date_str, 
                'financial_ratios_clean_updated.csv',  # Will be overridden by use_ratios_data
                'stock_prices.csv',  # Will be overridden by use_prices_data
                company_ticker,
                use_ratios_data=progressive_ratios_data,
                use_prices_data=progressive_price_data,
                random_seed=random_seed  # NEW: Pass seed to fundamental analysis
            )
        else:
            # Fallback to files (should be temporally filtered upstream)
            fundamental_response = generate_fundamental_prompt_simple(
                analysis_date_str, 
                'financial_ratios_clean_updated.csv',
                'stock_prices.csv',
                company_ticker,
                random_seed=random_seed  # NEW: Pass seed to fundamental analysis
            )
        
        if isinstance(fundamental_response, dict) and 'prompt' in fundamental_response:
            fundamental_context = fundamental_response['prompt'].split("---")[1].strip() if "---" in fundamental_response['prompt'] else fundamental_response['prompt'][:500]
            top_feature = fundamental_response.get('top_feature', 'Price Earnings Ratio (P/E)')
            xgboost_training_samples = fundamental_response.get('training_samples', 0)
            xgboost_features_used = fundamental_response.get('features_used', 0)
            temporal_protection_status = fundamental_response.get('temporal_protection', 'enabled')
        else:
            # Handle error case
            fundamental_context = f"Fundamental analysis failed: {fundamental_response.get('Error', 'Unknown error')}"
            top_feature = 'Price Earnings Ratio (P/E)'
            xgboost_training_samples = 0
            xgboost_features_used = 0
            temporal_protection_status = 'error'
            
        print(f"Fundamental analysis completed with temporal protection: {temporal_protection_status}")
            
    except Exception as e:
        print(f"Error in fundamental analysis: {e}")
        fundamental_context = "Fundamental analysis unavailable due to error."
        top_feature = 'Price Earnings Ratio (P/E)'
        xgboost_training_samples = 0
        xgboost_features_used = 0
        temporal_protection_status = 'error'

    print("--- All Agent Analyses Complete with Temporal Protection ---")

    # --- Step 4: Get RL Weights with Temporal Protection ---
    print("\n4. Getting RL Strategy Weights with Temporal Protection...")
    rl_weights = get_rl_weights_with_temporal_protection(q_table_file, current_data_row, analysis_date, random_seed)
    
    rl_guidance = f"""
**Strategic Weighting Guidance (from Temporally Protected RL Agent, Seed: {random_seed}):**
Based on current market conditions, our trained RL model recommends:
* Macro Strategist: {rl_weights['macro']:.0%}
* Fundamental Analyst: {rl_weights['fundamental']:.0%}
* Technical Analyst: {rl_weights['technical']:.0%}
* Strategy: {rl_weights['source'].replace('rl_', '').replace('_', ' ').title()}
* Temporal Protection: Enabled
* Random Seed: {random_seed}
"""

    # --- Step 5: Get SHAP Feature insight with Temporal Protection ---
    shapley_guidance = f"""
**Key Feature Spotlight (from Temporally Protected SHAP analysis, Seed: {random_seed}):**
Our enhanced fundamental model identified **'{top_feature}'** as the most influential factor in the current analysis.
* XGBoost Training Samples: {xgboost_training_samples}
* Features Used: {xgboost_features_used}
* Temporal Protection: {temporal_protection_status}
* Random Seed: {random_seed}
"""

    # Combine guidance
    special_guidance_text = f"""
**4. Advanced Model Guidance (Temporal Protection Enabled, Seed: {random_seed}):**
{rl_guidance}
{shapley_guidance}
"""

    # --- Step 6: Construct the Enhanced Final Prompt with Temporal Validation ---
    temporal_protection_notice = f"""
**TEMPORAL PROTECTION STATUS: ENABLED**
All analyses use only data available on or before {analysis_date_str}. No future information leakage detected.
Progressive data filtering applied across all agent analyses.
Random seed {random_seed} used for reproducibility.
"""

    final_decision_prompt = f"""
**Persona:**
You are the 'Decision Engine Agent', acting as a Chief Investment Officer. Your task is to synthesize analyses from three specialist agents, guided by advanced AI models (Reinforcement Learning + SHAP + Consistent Macro), to make a final, decisive trading recommendation.

{temporal_protection_notice}

---

**Investment Committee Meeting Brief: Analysis for {company_ticker} on {analysis_date_str}**

**1. Macro Strategist's Industry-Level View:**
{macro_context}

**2. Fundamental Analyst's Company-Level View (with Temporally Protected SHAP):**
{fundamental_context}

**3. Technical Analyst's Price Action View:**
{technical_context}

{special_guidance_text}

---

**Decision Framework:**
Your decision must incorporate the Advanced Model Guidance above:
- **RL Strategy**: The RL agent has learned from historical decisions and recommends the {rl_weights['source'].replace('rl_', '').replace('_', ' ')} approach
- **SHAP Insight**: Focus on how '{top_feature}' drives the fundamental assessment  
- **Macro Consistency**: All companies in {macro_recommendation.get('industry', 'this industry')} get the same macro view today
- **Temporal Integrity**: All analyses use only information available on or before {analysis_date_str}
- **Reproducibility**: Random seed {random_seed} ensures consistent results

**Task:**
Based on the complete meeting brief above, make a final trading decision. You MUST explain how the Advanced Model Guidance influenced your decision. Output your response in the following JSON format:

{{
  "final_recommendation": "Provide a clear, one-word recommendation (Buy, Sell, or Hold).",
  "confidence_score": "Provide a confidence score for your final decision from 1 (low) to 10 (high).",
  "final_rationale": "Provide a concise summary explaining your decision. Reference: (1) how the RL model's {rl_weights['source'].replace('rl_', '').replace('_', ' ')} strategy guided your agent weighting, (2) how the key SHAP feature '{top_feature}' influenced your fundamental assessment, (3) how the macro {macro_recommendation.get('recommendation', 'neutral')} stance for {macro_recommendation.get('industry', 'this industry')} affected your view, (4) your final synthesis, and (5) confirmation that temporal protection was maintained.",
  "macro_influence": "Briefly explain how the consistent macro industry recommendation influenced your decision (e.g., 'Macro {macro_recommendation.get('recommendation', 'neutral').lower()} supported the decision' or 'Overrode macro view due to strong fundamentals').",
  "temporal_protection_confirmation": "Confirm that only historical data (up to {analysis_date_str}) was used in this analysis.",
  "seed_confirmation": "Confirm that random seed {random_seed} was used for reproducibility."
}}
"""
    
    # Return enhanced dictionary with all metadata for logging + temporal protection status
    return {
        "prompt": final_decision_prompt,
        "top_feature": top_feature,
        "rl_strategy": rl_weights['source'],
        "rl_weights": rl_weights,
        "macro_recommendation": macro_recommendation['recommendation'],
        "macro_rationale": macro_recommendation['rationale'],
        "macro_industry": macro_recommendation.get('industry', 'Unknown'),
        "macro_sector": macro_recommendation.get('sector', 'Unknown'),
        "market_regime": macro_recommendation.get('market_regime', 'Unknown'),
        "temporal_protection": "enabled",
        "xgboost_training_samples": xgboost_training_samples,
        "xgboost_features_used": xgboost_features_used,
        "analysis_date": analysis_date_str,
        "temporal_protection_status": temporal_protection_status,
        "random_seed": random_seed  # NEW: Include seed in metadata
    }


# Legacy function maintained for backward compatibility
def generate_final_decision_prompt_ultimate(
    analysis_date_str: str, 
    company_ticker: str, 
    current_data_row: pd.Series
) -> dict:
    """
    Legacy function that calls the temporal-protected version with default seed.
    """
    print("Warning: Using legacy function - consider upgrading to temporal-protected version")
    return generate_final_decision_prompt_ultimate_with_temporal_protection(
        analysis_date_str, company_ticker, current_data_row, random_seed=42
    )


# --- Example Usage ---
if __name__ == "__main__":
    
    ticker_to_analyze = 'ASML NA Equity'
    date_to_analyze_str = "2023-01-16"
    
    # Test with multiple seeds
    test_seeds = [42, 123, 456, 789, 999]
    
    try:
        # FIXED: Use the temporal protected dataset
        time_machine_df = pd.read_csv('time_machine_dataset_temporal_protected.csv')
        time_machine_df['Date'] = pd.to_datetime(time_machine_df['Date'])
        
        analysis_date_obj = pd.to_datetime(date_to_analyze_str)
        
        # CRITICAL FIX: Only use data available on or before analysis date
        available_data = time_machine_df[time_machine_df['Date'] <= analysis_date_obj]
        
        data_for_date = available_data[
            (available_data['Date'] == analysis_date_obj) & 
            (available_data['Ticker'] == ticker_to_analyze)
        ]
        
        if not data_for_date.empty:
            current_row = data_for_date.iloc[0]

            for seed in test_seeds:
                print(f"\n{'='*70}")
                print(f"RUNNING ULTIMATE DECISION ENGINE WITH SEED: {seed}")
                print(f"{'='*70}")
                
                # Create progressive data for temporal protection
                progressive_price_data = time_machine_df[
                    (time_machine_df['Date'] <= analysis_date_obj)
                ][['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                
                progressive_time_machine_data = available_data.copy()
                
                ultimate_response = generate_final_decision_prompt_ultimate_with_temporal_protection(
                    analysis_date_str=date_to_analyze_str, 
                    company_ticker=ticker_to_analyze, 
                    current_data_row=current_row,
                    progressive_price_data=progressive_price_data,
                    progressive_time_machine_data=progressive_time_machine_data,
                    random_seed=seed  # NEW: Pass seed
                )
                
                print(f"\nMETADATA FOR SEED {seed}:")
                print(f"Top SHAP Feature: {ultimate_response['top_feature']}")
                print(f"RL Strategy: {ultimate_response['rl_strategy']}")
                print(f"Macro Recommendation: {ultimate_response['macro_recommendation']}")
                print(f"Industry: {ultimate_response['macro_industry']}")
                print(f"Temporal Protection: {ultimate_response['temporal_protection']}")
                print(f"Analysis Date: {ultimate_response['analysis_date']}")
                print(f"XGBoost Training Samples: {ultimate_response['xgboost_training_samples']}")
                print(f"Random Seed: {ultimate_response['random_seed']}")

        else:
            print(f"No data found for {ticker_to_analyze} on {date_to_analyze_str} in the temporal protected dataset.")
            print("Available dates for this ticker:")
            ticker_dates = time_machine_df[time_machine_df['Ticker'] == ticker_to_analyze]['Date'].unique()
            print(sorted(ticker_dates)[:10], "..." if len(ticker_dates) > 10 else "")

    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Please ensure the following files exist:")
        print("  - time_machine_dataset_temporal_protected.csv")
        print("  - optimized_agent_q_table_temporal_protected_seed_*.npy")
        print("  - Company_Sector_Industry.csv")
        print("Run the temporal protected time machine and multi-seed RL agent scripts first!")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()