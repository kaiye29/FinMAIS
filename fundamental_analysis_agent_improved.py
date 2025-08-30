import pandas as pd
from datetime import datetime
import xgboost as xgb
import shap
import warnings
import random
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def train_xgboost_with_temporal_split(merged_df, analysis_date, min_training_days=365):
    """
    Ensures XGBoost training uses proper temporal splits.
    """
    # Calculate training cutoff (use 80% of available historical data for training)
    earliest_date = merged_df.index.min()
    latest_available = analysis_date - pd.DateOffset(days=30)  # Already filtered
    
    total_days = (latest_available - earliest_date).days
    training_days = max(min_training_days, int(total_days * 0.8))
    
    training_cutoff = latest_available - pd.DateOffset(days=int(total_days * 0.2))
    
    # Split data temporally
    train_data = merged_df[merged_df.index <= training_cutoff]
    validation_data = merged_df[merged_df.index > training_cutoff]
    
    print(f"Temporal split: Training up to {training_cutoff.date()}, Validation after")
    print(f"Training samples: {len(train_data)}, Validation samples: {len(validation_data)}")
    
    return train_data, validation_data

def generate_fundamental_prompt_simple(
    analysis_date_str: str,
    ratios_filepath: str,
    prices_filepath: str,
    company_ticker: str,
    use_ratios_data=None,
    use_prices_data=None,
    random_seed=42  # NEW: Add random seed parameter
) -> dict:
    """
    FIXED: Simplified version with proper temporal data protection and random seed control.
    Extracts ratios first, then builds the model using only historical data.
    """
    try:
        # Set random seeds for reproducibility
        set_random_seeds(random_seed)
        
        analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
        
        # --- CRITICAL FIX: Load data with temporal filtering ---
        if use_ratios_data is not None and use_prices_data is not None:
            # Use pre-filtered data (from backtest with temporal protection)
            ratios_df = use_ratios_data.copy()
            price_df = use_prices_data.copy()
            print(f"Using temporally filtered data (no future leakage)")
        else:
            # Load from files and apply temporal filtering
            ratios_df = pd.read_csv(ratios_filepath, parse_dates=['Date'])
            price_df = pd.read_csv(prices_filepath, parse_dates=['Date'])
            
            # CRITICAL: Only use data available on or before analysis date
            ratios_df = ratios_df[ratios_df['Date'] <= analysis_date].copy()
            price_df = price_df[price_df['Date'] <= analysis_date].copy()
            print(f"Applied temporal filtering: no data after {analysis_date_str}")
        
        # Validate temporal protection
        if len(ratios_df) > 0 and ratios_df['Date'].max() > analysis_date:
            raise ValueError("TEMPORAL VIOLATION: Ratios data contains future information!")
        if len(price_df) > 0 and price_df['Date'].max() > analysis_date:
            raise ValueError("TEMPORAL VIOLATION: Price data contains future information!")
        
        # --- STEP 1: Get current ratios (using the working baseline approach) ---
        ticker_ratios = ratios_df[
            (ratios_df['Ticker'] == company_ticker) & 
            (ratios_df['Date'] <= analysis_date)
        ]
        
        if ticker_ratios.empty:
            return {"Error": f"No ratios data found for {company_ticker} on or before {analysis_date_str}"}
        
        # Get the most recent ratios
        latest_date = ticker_ratios['Date'].max()
        latest_ratios = ticker_ratios[ticker_ratios['Date'] == latest_date]
        
        # Create ratios dictionary (this should work like your baseline)
        current_ratios = {}
        for _, row in latest_ratios.iterrows():
            current_ratios[row['Field']] = row['Value']
        
        # --- STEP 2: Build model data with temporal protection ---
        # Filter data for this ticker with strict temporal boundaries
        ratios_df_filtered = ratios_df[
            (ratios_df['Ticker'] == company_ticker) & 
            (ratios_df['Date'] <= analysis_date)
        ].copy()
        
        price_df_filtered = price_df[
            (price_df['Ticker'] == company_ticker) & 
            (price_df['Date'] <= analysis_date)
        ].copy()
        
        if ratios_df_filtered.empty or price_df_filtered.empty:
            return {"Error": f"Insufficient historical data for {company_ticker} before {analysis_date_str}"}
        
        # Create wide format with temporal protection
        ratios_wide = ratios_df_filtered.pivot(index='Date', columns='Field', values='Value')
        merged_df = pd.merge(ratios_wide, price_df_filtered[['Date', 'Close']], on='Date', how='inner')
        merged_df.set_index('Date', inplace=True)
        merged_df.sort_index(inplace=True)
        
        # CRITICAL FIX: Feature engineering with temporal protection
        # Only calculate future returns for dates where we can safely look ahead
        # within the allowed historical period
        merged_df['next_month_close'] = merged_df['Close'].shift(-21)
        merged_df['target'] = (merged_df['next_month_close'] > merged_df['Close']).astype(int)
        
        # Remove rows where future returns would require data beyond analysis_date
        cutoff_date = analysis_date - pd.DateOffset(days=30)  # Ensure we don't peek into future
        merged_df = merged_df[merged_df.index <= cutoff_date]
        
        merged_df.dropna(subset=['Close', 'next_month_close'], inplace=True)
        
        if len(merged_df) < 10:  # Need minimum data for training
            return {"Error": f"Insufficient training data for {company_ticker} (only {len(merged_df)} samples)"}
        
        # --- ENHANCED: Use temporal split for training ---
        try:
            train_data, validation_data = train_xgboost_with_temporal_split(merged_df, analysis_date)
            
            if len(train_data) < 10:
                # Fall back to using all data if temporal split results in too little training data
                print("Warning: Temporal split resulted in insufficient training data, using all available data")
                train_data = merged_df
                
        except Exception as e:
            print(f"Warning: Temporal split failed: {e}, using all available data")
            train_data = merged_df
        
        features = train_data.drop(columns=['Close', 'next_month_close', 'target'])
        target = train_data['target']
        features = features.apply(pd.to_numeric, errors='coerce')
        
        # --- STEP 3: Train model with temporal protection AND seed control ---
        # Use temporally split training data
        train_features = features.copy()
        train_target = target.copy()
        
        # Additional validation
        if train_features.empty or train_target.nunique() < 2:
            return {"Error": "Insufficient training data or no target variation"}
        
        # Drop columns with too many NaN values
        train_features = train_features.dropna(axis=1, thresh=int(0.7 * len(train_features)))
        train_features = train_features.fillna(train_features.median())
        
        if train_features.shape[1] == 0:
            return {"Error": "No valid features after cleaning"}
        
        print(f"Training XGBoost on {len(train_features)} samples with {train_features.shape[1]} features (seed: {random_seed})")
        
        # UPDATED: Use the passed random seed
        model = xgb.XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss',
            n_estimators=50, 
            max_depth=3, 
            random_state=random_seed,  # Use parameter instead of hardcoded 42
            verbosity=0  # Suppress training output
        )
        
        try:
            model.fit(train_features, train_target)
        except Exception as e:
            return {"Error": f"XGBoost training failed: {str(e)}"}
        
        # --- STEP 4: SHAP analysis with temporal protection ---
        # Find the most recent row for analysis (within temporal boundaries)
        analysis_features = train_features.copy()
        
        if analysis_features.empty:
            return {"Error": "No features available for SHAP analysis"}
        
        # Use the most recent available data point
        latest_model_row = analysis_features.tail(1)
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(latest_model_row)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            shap_df = pd.DataFrame(shap_values, columns=analysis_features.columns, index=latest_model_row.index)
            contrib = shap_df.iloc[0]
            
            sorted_contrib = contrib.sort_values()
            most_negative_feature = sorted_contrib.index[0]
            most_negative_value = sorted_contrib.iloc[0]
            most_positive_feature = sorted_contrib.index[-1]  
            most_positive_value = sorted_contrib.iloc[-1]
            most_impactful_feature = contrib.abs().idxmax()
            
            shap_analysis = (
                f"* The most significant positive contributor to the prediction was: **{most_positive_feature} ({most_positive_value:.2f})**\n"
                f"* The most significant negative contributor to the prediction was: **{most_negative_feature} ({most_negative_value:.2f})**"
            )
            
            print(f"SHAP analysis successful. Most impactful: {most_impactful_feature}")
            
        except Exception as e:
            print(f"Warning: SHAP analysis failed: {e}")
            # Fallback to feature importance
            feature_importance = model.feature_importances_
            most_important_idx = feature_importance.argmax()
            most_impactful_feature = analysis_features.columns[most_important_idx]
            shap_analysis = f"* Feature importance analysis identified **{most_impactful_feature}** as the most impactful factor."
        
        # --- STEP 5: Extract the 10 ratios (using current_ratios dict) ---
        def safe_get_ratio(field_name):
            val = current_ratios.get(field_name)
            if val is None or pd.isna(val):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        # The exact field names from baseline
        pe_ratio = safe_get_ratio('Price Earnings Ratio (P/E)')
        pb_ratio = safe_get_ratio('Price to Book Ratio')
        dividend_yield = safe_get_ratio('Dividend 12 Month Yield')
        profit_margin = safe_get_ratio('Profit Margin')
        roe = safe_get_ratio('Return on Common Equity')
        sales_growth = safe_get_ratio('Revenue Growth Year over Year')
        current_ratio = safe_get_ratio('Current Ratio')
        debt_to_equity = safe_get_ratio('Total Debt to Total Equity')
        fcf_to_debt = safe_get_ratio('Free Cash Flow to Total Debt')
        inventory_turnover = safe_get_ratio('Inventory Turnover')
        
        def fmt(val, is_percent=False):
            if val is not None:
                return f"{val:.2f}{'%' if is_percent else ''}"
            return "N/A"
        
        # Format strings
        pe_ratio_str = fmt(pe_ratio)
        pb_ratio_str = fmt(pb_ratio)
        dividend_yield_str = fmt(dividend_yield, is_percent=True)
        profit_margin_str = fmt(profit_margin, is_percent=True)
        roe_str = fmt(roe, is_percent=True)
        sales_growth_str = fmt(sales_growth, is_percent=True)
        current_ratio_str = fmt(current_ratio)
        debt_to_equity_str = fmt(debt_to_equity)
        fcf_to_debt_str = fmt(fcf_to_debt)
        inventory_turnover_str = fmt(inventory_turnover)
        
        # --- STEP 6: Build final prompt with temporal validation ---
        final_prompt = f"""
**Persona:**
You are a 'Fundamental Analyst Agent'. Your task is to analyze a stock's key financial ratios to determine its valuation. Your reasoning MUST be grounded in the verifiable evidence provided. All analysis uses only historical data to prevent future information leakage.

---

**Context: Fundamental Data for {company_ticker} as of {latest_date.date()} (Temporal Protection: Enabled, Seed: {random_seed})**

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

**2. XGBoost + SHAP Analysis (Temporal Protection Enabled, Seed: {random_seed}):**
A machine learning model predicted the stock's direction based on historical data only (no future information). The analysis below shows which ratios had the biggest impact on that prediction:
{shap_analysis}

**Data Integrity:** No future data leakage - all analysis uses only information available on or before {analysis_date_str}

---

**Task:**
Based strictly on the comprehensive ratios and the temporal-protected XGBoost/SHAP analysis provided, generate a structured analysis. Your rationale **must** reference the model analysis to justify your conclusion. Output your response in the following JSON format:

{{
  "valuation_assessment": "Provide the overall valuation assessment (e.g., 'Undervalued', 'Fairly Valued', 'Overvalued', 'High Growth Potential but Risky').",
  "confidence_score": "Provide a confidence score from 1 (low) to 10 (high).",
  "rationale": "Provide a concise, one-paragraph summary explaining how the ratios and **especially the XGBoost/SHAP analysis** support your conclusion. Reference the most impactful features identified by the model and explain how they align with the traditional ratio analysis."
}}
"""
        
        return {
            "prompt": final_prompt,
            "top_feature": most_impactful_feature,
            "temporal_protection": "enabled",
            "analysis_date": analysis_date_str,
            "training_samples": len(train_features),
            "features_used": train_features.shape[1],
            "random_seed": random_seed  # NEW: Include seed in metadata
        }
        
    except Exception as e:
        return {"Error": f"Error in temporal-protected analysis: {str(e)}"}

# --- Example Usage ---
if __name__ == "__main__":
    print("TESTING IMPROVED FUNDAMENTAL AGENT WITH ENHANCED TEMPORAL PROTECTION AND SEED CONTROL")
    print("="*60)
    
    ratios_file = 'financial_ratios_clean_updated.csv'
    prices_file = 'stock_prices.csv'
    
    ticker = 'ASML NA Equity'
    date = "2020-05-20"  # Use a date with sufficient historical data
    
    # Test with multiple seeds
    test_seeds = [42, 123, 456]
    
    for seed in test_seeds:
        print(f"\n--- Testing with seed {seed} ---")
        
        res = generate_fundamental_prompt_simple(
            date, ratios_file, prices_file, ticker, random_seed=seed
        )
        
        if "Error" in res:
            print(f"ERROR: {res['Error']}")
        else:
            print(f"Success! Top feature: {res.get('top_feature', 'N/A')}")
            print(f"Training samples: {res.get('training_samples', 'N/A')}")
            print(f"Seed used: {res.get('random_seed', 'N/A')}")
            print(f"Features used: {res.get('features_used', 'N/A')}")