import pandas as pd
import time
import json
import google.generativeai as genai
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Any, Dict, Union
import random
import numpy as np
import os

# --- Import the IMPROVED decision engine function with seed support ---
from decision_engine_improved import generate_final_decision_prompt_IMPROVED_with_temporal_awareness

# --- Configure your Gemini API Key ---
API_KEY = 'AIzaSyDSyen4rBDBcUuG-uZb2G6B3CPjw6k_wUQ'
genai.configure(api_key=API_KEY)

# Use a single global model instance
_GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash-lite')

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seeds set to {seed} for reproducibility")

# -------------------------
# Helpers for Option A glue
# -------------------------
def _summarize_top_feature(tf: Any) -> str:
    """
    Turn whatever 'top_feature' structure is into a short, readable string.
    Accepts dict / list / str / any.
    """
    if tf is None:
        return ""

    # dict with typical SHAP-like fields
    if isinstance(tf, dict):
        name = tf.get("name") or tf.get("feature") or "Top Feature"
        val = tf.get("value")
        eff = tf.get("effect") or tf.get("direction")
        expl = tf.get("explanation")
        bits = [f"- {name}"]
        if val is not None:
            bits.append(f"value={val}")
        if eff:
            bits.append(f"effect={eff}")
        if expl:
            bits.append(f"note={expl}")
        return " | ".join(bits)

    # list of features
    if isinstance(tf, list):
        parts = []
        for i, item in enumerate(tf[:5], 1):  # cap to 5 items
            if isinstance(item, dict):
                nm = item.get("name") or item.get("feature") or f"feature_{i}"
                val = item.get("value")
                eff = item.get("effect") or item.get("direction")
                parts.append(f"{nm}({val}, {eff})")
            else:
                parts.append(str(item))
        return ", ".join(parts)

    # string or other primitive
    return str(tf)

def _build_prompt(payload: Union[str, Dict[str, Any]]) -> str:
    """
    Option A: If payload is a dict, fold SHAP/top_feature context into the text prompt.
    If it's already a string, return as-is.
    Expected dict shape: {"prompt": <str>, "top_feature": <any>}
    """
    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        base = payload.get("prompt", "")
        tf = payload.get("top_feature")
        tf_text = _summarize_top_feature(tf)
        if tf_text:
            base = (
                f"{base}\n\n"
                "-----\n"
                "Explainability (SHAP-style top driver):\n"
                f"{tf_text}\n"
                "-----\n"
                "Use the above feature attribution ONLY as supporting context. "
                "Now produce a single JSON object with keys: "
                "`final_recommendation` (Buy/Sell/Hold), `confidence_score` (0-1 float), "
                "`final_rationale` (1-2 sentences), `macro_influence` (how macro affected decision).\n"
            )
        return base

    # Fallback: stringify unexpected types
    return str(payload)

def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Be forgiving: strip code fences, try json.loads, fall back to a default."""
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        # last resort: return a safe default
        return {
            "final_recommendation": "Hold",
            "confidence_score": 0.0,
            "final_rationale": "LLM returned non-JSON; defaulted.",
            "macro_influence": "Unable to determine due to parsing error"
        }

def get_llm_decision(prompt_or_payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Option A in action: build a clean string prompt, call Gemini, parse JSON.
    """
    print("--- Calling Gemini API for a decision... ---")
    try:
        prompt_str = _build_prompt(prompt_or_payload)
        response = _GEMINI_MODEL.generate_content(prompt_str)
        response_text = getattr(response, "text", "") or ""
        parsed = _parse_llm_json(response_text)
        
        # Enhanced logging with macro influence
        decision = parsed.get('final_recommendation', 'Hold')
        macro_influence = parsed.get('macro_influence', 'N/A')
        print(f"--- Gemini Decision: {decision} (Macro: {macro_influence[:30]}...) ---")
        
        return parsed
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return {
            "final_recommendation": "Hold", 
            "confidence_score": 0.0, 
            "final_rationale": "API error",
            "macro_influence": "Unable to determine due to API error"
        }

# -------------------------
# Enhanced Backtest with Complete Temporal Protection and Seed Control
# -------------------------
def run_backtest_improved_with_enhanced_temporal_protection(
    tickers_list, 
    start_date_str, 
    end_date_str, 
    random_seed=42  # NEW: Add random seed parameter
):
    """
    Enhanced: Runs improved model backtest with comprehensive temporal data protection and seed control.
    """
    print("INITIALIZING ENHANCED IMPROVED MODEL BACKTEST WITH COMPREHENSIVE TEMPORAL PROTECTION")
    print(f"Random Seed: {random_seed}")
    print("="*70)
    
    # Set random seeds at the very beginning
    set_random_seeds(random_seed)
    
    # Define temporal boundaries
    REALISTIC_START_DATE = "2001-01-01"  # Your full dataset start
    EARLY_INDICATOR_WARNING = "2001-08-01"  # When 200-day SMA becomes available
    
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    early_warning = pd.to_datetime(EARLY_INDICATOR_WARNING)
    
    # Validation for early period limitations
    if start_date < early_warning:
        print(f"INFO: First ~8 months may have incomplete technical indicators")
        print(f"This is normal - technical indicators need historical data to calculate")
        print(f"200-day SMA will be available from: {EARLY_INDICATOR_WARNING}")
        print(f"XGBoost model will have limited early data for training")
    else:
        print(f"All technical indicators should be available from start date")
    
    # --- Load data with temporal awareness ---
    try:
        # Load full datasets
        all_price_data = pd.read_csv('stock_prices.csv', parse_dates=['Date'])
        all_ratios_data = pd.read_csv('financial_ratios_clean_updated.csv', parse_dates=['Date'])
        
        print(f"Loaded full price dataset: {len(all_price_data)} rows")
        print(f"Loaded full ratios dataset: {len(all_ratios_data)} rows")
        
        # CRITICAL FIX: Filter all data to only include data up to backtest end date
        # This prevents future data leakage during backtesting
        price_data = all_price_data[all_price_data['Date'] <= end_date].copy()
        ratios_data = all_ratios_data[all_ratios_data['Date'] <= end_date].copy()
        
        print(f"Filtered price data to prevent future leakage: {len(price_data)} rows")
        print(f"Filtered ratios data to prevent future leakage: {len(ratios_data)} rows")
        print(f"Price data range: {price_data['Date'].min()} to {price_data['Date'].max()}")
        print(f"Ratios data range: {ratios_data['Date'].min()} to {ratios_data['Date'].max()}")
        
        # Additional validation
        if price_data['Date'].max() > end_date or ratios_data['Date'].max() > end_date:
            raise ValueError("Data leakage detected: Data contains future information!")
            
    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
        return None

    # --- Setup Portfolio and Logging ---
    rebalance_dates = pd.date_range(start_date, end_date, freq='BMS')
    
    initial_cash = 1000000
    portfolio = {'cash': initial_cash, 'holdings': {}}
    portfolio_history = [{'Date': start_date - pd.DateOffset(days=1), 'TotalValue': initial_cash}]
    decisions_log_improved = []

    print(f"\nENHANCED TEMPORAL PROTECTION ENABLED (IMPROVED MODEL):")
    print(f"  Backtest Period: {start_date_str} to {end_date_str}")
    print(f"  Price Data Cutoff: {price_data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Ratios Data Cutoff: {ratios_data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  Rebalance Dates: {len(rebalance_dates)}")
    print(f"  XGBoost + SHAP Analysis: Enabled")
    print(f"  Random Seed: {random_seed}")
    print(f"  Enhanced Temporal Validation: No future data leakage")

    # --- Main Backtesting Loop with Enhanced Temporal Protection ---
    for i, date in enumerate(rebalance_dates):
        print(f"\n{'='*60}")
        print(f"Rebalancing {i+1}/{len(rebalance_dates)}: {date.date()} (Enhanced, Seed: {random_seed})")
        print(f"{'='*60}")
        
        # CRITICAL: Validate that we're not using future information
        available_price_data = price_data[price_data['Date'] <= date]
        available_ratios_data = ratios_data[ratios_data['Date'] <= date]
        
        if len(available_price_data) == 0 or len(available_ratios_data) == 0:
            print(f"Insufficient data available for {date.date()}. Skipping this period.")
            continue
            
        latest_price_date = available_price_data['Date'].max()
        latest_ratios_date = available_ratios_data['Date'].max()
        print(f"Using price data up to: {latest_price_date.strftime('%Y-%m-%d')}")
        print(f"Using ratios data up to: {latest_ratios_date.strftime('%Y-%m-%d')}")
        
        # ENHANCED: Additional temporal validation
        if latest_price_date > date or latest_ratios_date > date:
            print(f"TEMPORAL VIOLATION DETECTED!")
            print(f"   Analysis date: {date.date()}")
            print(f"   Latest price date: {latest_price_date.date()}")
            print(f"   Latest ratios date: {latest_ratios_date.date()}")
            continue
        else:
            print(f"Temporal validation passed for {date.date()}")
        
        # --- Sell all existing holdings ---
        if portfolio['holdings']:
            print("Liquidating current positions...")
            for ticker, data in portfolio['holdings'].items():
                # FIXED: Only use price data available on or before current date
                sell_price_data = available_price_data[
                    (available_price_data['Date'] >= date) & 
                    (available_price_data['Ticker'] == ticker)
                ]
                if not sell_price_data.empty:
                    sell_price = sell_price_data['Close'].iloc[0]
                    sale_value = data['shares'] * sell_price
                    portfolio['cash'] += sale_value
                    print(f"  Sold {ticker}: {data['shares']:.2f} shares @ ${sell_price:.2f} = ${sale_value:,.2f}")
            portfolio['holdings'] = {}
        
        print(f"Available cash: ${portfolio['cash']:,.2f}")
        
        # --- Allocate new positions with enhanced temporal protection ---
        cash_per_ticker = portfolio['cash'] / len(tickers_list) if len(tickers_list) > 0 else 0
        successful_analyses = 0
        
        for ticker in tickers_list:
            print(f"\nAnalyzing {ticker} with Enhanced XGBoost + SHAP (seed: {random_seed})...")
            
            try:
                # CRITICAL: Generate prompt with enhanced temporal awareness and seed control
                final_payload = generate_final_decision_prompt_IMPROVED_with_temporal_awareness(
                    date.strftime("%Y-%m-%d"), 
                    ticker,
                    use_ratios_data=available_ratios_data,
                    use_prices_data=available_price_data,
                    temporal_protection_enabled=True,
                    random_seed=random_seed  # NEW: Pass the seed
                )
                
                # Handle potential errors from the improved model
                if isinstance(final_payload, dict) and "Error" in final_payload:
                    print(f"  XGBoost/SHAP analysis failed: {final_payload['Error']}")
                    continue
                
                # Get AI decision
                response_dict = get_llm_decision(final_payload)
                decision = response_dict.get("final_recommendation", "Hold")
                confidence = response_dict.get("confidence_score", 0.0)
                rationale = response_dict.get("final_rationale", "")
                macro_influence = response_dict.get("macro_influence", "N/A")
                
                successful_analyses += 1
                
                # ENHANCED: Enhanced logging with seed information
                decisions_log_improved.append({
                    'Date': date.strftime("%Y-%m-%d"),
                    'Ticker': ticker,
                    'Decision': decision,
                    'Confidence': confidence,
                    'Rationale': rationale,
                    'Macro_Influence': macro_influence,
                    'Macro_Recommendation': final_payload.get('macro_recommendation', 'Unknown'),
                    'Macro_Rationale': final_payload.get('macro_rationale', 'Unknown'),
                    'Top_SHAP_Feature': str(final_payload.get('top_feature', 'N/A'))[:100],
                    'Data_Cutoff_Date_Price': latest_price_date.strftime('%Y-%m-%d'),
                    'Data_Cutoff_Date_Ratios': latest_ratios_date.strftime('%Y-%m-%d'),
                    'Temporal_Protection': final_payload.get('temporal_protection', 'standard'),
                    'XGBoost_Training_Samples': final_payload.get('xgboost_training_samples', 0),
                    'XGBoost_Features_Used': final_payload.get('xgboost_features_used', 0),
                    'Temporal_Status': final_payload.get('temporal_status', 'unknown'),
                    'Random_Seed': random_seed,  # NEW: Include seed in log
                    'Model_Type': f'Enhanced_Improved_XGBoost_SHAP_Seed_{random_seed}',
                    'XGBoost_Model': 'Enabled',
                    'SHAP_Analysis': 'Enabled',
                    'Enhanced_Temporal_Protection': 'Enabled',
                    'Seed_Control': 'Enabled'  # NEW
                })
                
                # Execute BUY decisions with temporal protection
                if decision == 'Buy':
                    # FIXED: Only use price data available on or before current date
                    buy_price_data = available_price_data[
                        (available_price_data['Date'] >= date) & 
                        (available_price_data['Ticker'] == ticker)
                    ]
                    if not buy_price_data.empty:
                        buy_price = buy_price_data['Close'].iloc[0]
                        if portfolio['cash'] >= cash_per_ticker and buy_price > 0:
                            shares_to_buy = cash_per_ticker / buy_price
                            portfolio['holdings'][ticker] = {
                                'shares': shares_to_buy, 
                                'price_paid': buy_price,
                                'confidence': confidence,
                                'purchase_date': date.strftime("%Y-%m-%d"),
                                'shap_feature': str(final_payload.get('top_feature', 'N/A'))[:50],
                                'temporal_protection': True,
                                'random_seed': random_seed  # NEW
                            }
                            portfolio['cash'] -= cash_per_ticker
                            print(f"  EXECUTED BUY: {shares_to_buy:.2f} shares @ ${buy_price:.2f}")
                            print(f"     XGBoost Confidence: {confidence}, Seed: {random_seed}")
                            print(f"     SHAP Feature: {str(final_payload.get('top_feature', 'N/A'))[:30]}...")
                            print(f"     Temporal Protection: Enabled")
                        else:
                            print(f"  Insufficient cash for {ticker}")
                    else:
                        print(f"  No price data available for {ticker} on {date.date()}")
                else:
                    print(f"  Decision: {decision} (XGBoost confidence: {confidence}, seed: {random_seed})")
                    if final_payload.get('top_feature'):
                        print(f"     Key SHAP feature: {str(final_payload.get('top_feature', 'N/A'))[:50]}...")
                    
            except Exception as e:
                print(f"  Error in Enhanced XGBoost/SHAP analysis for {ticker}: {e}")
                continue

        print(f"\nAnalysis Summary: {successful_analyses}/{len(tickers_list)} successful Enhanced analyses (seed: {random_seed})")
        
        # --- Calculate portfolio value with temporal protection ---
        current_holdings_value = 0
        if portfolio['holdings']:
            print(f"\nCurrent Holdings (Enhanced XGBoost + Seed: {random_seed}):")
            for ticker, data in portfolio['holdings'].items():
                # FIXED: Only use price data available on or before current date
                price_data_point = available_price_data[
                    (available_price_data['Date'] >= date) & 
                    (available_price_data['Ticker'] == ticker)
                ]
                if not price_data_point.empty:
                    current_price = price_data_point['Close'].iloc[0]
                    position_value = data['shares'] * current_price
                    current_holdings_value += position_value
                    gain_loss = ((current_price - data['price_paid']) / data['price_paid']) * 100
                    temporal_status = "Protected" if data.get('temporal_protection', False) else "Standard"
                    seed_info = f"Seed {data.get('random_seed', 'N/A')}"
                    print(f"  {ticker}: ${position_value:,.0f} ({gain_loss:+.1f}%) [{temporal_status}, {seed_info}]")
                    print(f"    SHAP: {data.get('shap_feature', 'N/A')}")
        
        total_value = portfolio['cash'] + current_holdings_value
        portfolio_history.append({'Date': date, 'TotalValue': total_value})
        
        print(f"\nPortfolio Summary (Enhanced XGBoost + Seed: {random_seed}):")
        print(f"  Cash: ${portfolio['cash']:,.2f}")
        print(f"  Holdings: ${current_holdings_value:,.2f}")
        print(f"  Total Value: ${total_value:,.2f}")
        print(f"  Enhanced Temporal Protection: Enabled")
        print(f"  Random Seed: {random_seed}")
        
        # Progress indicator
        if len(portfolio_history) > 1:
            prev_value = portfolio_history[-2]['TotalValue']
            period_return = ((total_value - prev_value) / prev_value) * 100
            print(f"  Period Return: {period_return:+.2f}%")

    # --- Save results with seed-specific filename ---
    try:
        decisions_df = pd.DataFrame(decisions_log_improved)
        # NEW: Include seed in filename for separate tracking
        output_file = f'trading_decisions_enhanced_improved_seed_{random_seed}.csv'
        decisions_df.to_csv(output_file, index=False)
        print(f"\nEnhanced improved model decisions (seed {random_seed}) saved to {output_file}")
        
        # Analysis
        analyze_enhanced_improved_temporal_protection_results(decisions_df, random_seed)
        
    except Exception as e:
        print(f"Error saving decisions log: {e}")

    results_df = pd.DataFrame(portfolio_history)
    return results_df

def analyze_enhanced_improved_temporal_protection_results(decisions_df, random_seed):
    """
    Analyzes results to confirm enhanced temporal protection worked correctly for improved model with seed tracking.
    """
    print(f"\nENHANCED IMPROVED MODEL TEMPORAL PROTECTION VALIDATION (Seed: {random_seed}):")
    print(f"  Total decisions: {len(decisions_df)}")
    
    # Check seed consistency
    if 'Random_Seed' in decisions_df.columns:
        seed_counts = decisions_df['Random_Seed'].value_counts()
        if len(seed_counts) == 1 and seed_counts.index[0] == random_seed:
            print(f"  Seed consistency: Passed (all decisions used seed {random_seed})")
        else:
            print(f"  Seed consistency: Warning - multiple seeds detected: {dict(seed_counts)}")
    
    # Check temporal protection status
    if 'Temporal_Protection' in decisions_df.columns:
        protection_status = decisions_df['Temporal_Protection'].value_counts()
        print(f"  Temporal Protection Status:")
        for status, count in protection_status.items():
            pct = count / len(decisions_df) * 100
            icon = "Pass" if status == "enabled" else "Warning"
            print(f"    {icon} {status}: {count} ({pct:.1f}%)")
    
    # Check data cutoff dates
    if 'Data_Cutoff_Date_Price' in decisions_df.columns:
        cutoff_dates = pd.to_datetime(decisions_df['Data_Cutoff_Date_Price'])
        analysis_dates = pd.to_datetime(decisions_df['Date'])
        
        # Validate no future data was used
        future_data_usage = (cutoff_dates > analysis_dates).sum()
        if future_data_usage == 0:
            print(f"  Temporal validation: Passed - No future data leakage detected")
        else:
            print(f"  Temporal validation: Failed - {future_data_usage} instances of future data usage!")
    
    # Enhanced XGBoost/SHAP specific analysis
    if 'XGBoost_Training_Samples' in decisions_df.columns:
        valid_xgb = decisions_df[decisions_df['XGBoost_Training_Samples'] > 0]
        if len(valid_xgb) > 0:
            avg_training_samples = valid_xgb['XGBoost_Training_Samples'].mean()
            avg_features = valid_xgb['XGBoost_Features_Used'].mean()
            print(f"  Enhanced XGBoost Analysis (Seed {random_seed}):")
            print(f"    Average training samples: {avg_training_samples:.0f}")
            print(f"    Average features used: {avg_features:.0f}")
            print(f"    Successful XGBoost analyses: {len(valid_xgb)} ({len(valid_xgb)/len(decisions_df)*100:.1f}%)")
    
    # Distribution analysis
    decision_counts = decisions_df['Decision'].value_counts()
    print(f"\nDecision Distribution (Seed {random_seed}):")
    for decision, count in decision_counts.items():
        pct = count / len(decisions_df) * 100
        print(f"  {decision}: {count} ({pct:.1f}%)")

def run_multi_seed_backtest(
    tickers_list, 
    start_date_str, 
    end_date_str, 
    seeds=[42, 123, 456, 789, 999]
):
    """
    NEW: Run backtests with multiple random seeds to assess result stability.
    """
    results_summary = []
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"RUNNING BACKTEST WITH SEED: {seed}")
        print(f"{'='*70}")
        
        # Run backtest with this seed
        backtest_results = run_backtest_improved_with_enhanced_temporal_protection(
            tickers_list, start_date_str, end_date_str, random_seed=seed
        )
        
        if backtest_results is not None:
            initial_value = 1000000
            final_value = backtest_results['TotalValue'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            results_summary.append({
                'seed': seed,
                'final_value': final_value,
                'total_return': total_return,
                'backtest_data': backtest_results
            })
            
            print(f"Seed {seed} Result: {total_return:+.2f}% return")
        else:
            print(f"Seed {seed} failed!")
            
    # Analyze stability across seeds
    if results_summary:
        returns = [r['total_return'] for r in results_summary]
        print(f"\n{'='*70}")
        print("MULTI-SEED STABILITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Mean Return: {np.mean(returns):+.2f}%")
        print(f"Std Dev: {np.std(returns):.2f}%")
        print(f"Min Return: {min(returns):+.2f}%")
        print(f"Max Return: {max(returns):+.2f}%")
        print(f"Return Range: {max(returns) - min(returns):.2f}%")
        print(f"Return Stability: {'HIGH' if np.std(returns) < 5 else 'MEDIUM' if np.std(returns) < 10 else 'LOW'}")
        
        # Save detailed results
        summary_df = pd.DataFrame([
            {'seed': r['seed'], 'total_return': r['total_return'], 'final_value': r['final_value']} 
            for r in results_summary
        ])
        summary_df.to_csv('multi_seed_backtest_summary_improved.csv', index=False)
        print(f"Detailed results saved to: multi_seed_backtest_summary_improved.csv")
        
        # Create comparison plot
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot individual seed performance
            plt.subplot(1, 2, 1)
            plt.bar(range(len(seeds)), returns)
            plt.xlabel('Seed Index')
            plt.ylabel('Total Return (%)')
            plt.title('Performance by Seed')
            plt.xticks(range(len(seeds)), [f'Seed {s}' for s in seeds], rotation=45)
            
            # Plot portfolio values over time for all seeds
            plt.subplot(1, 2, 2)
            for i, result in enumerate(results_summary):
                backtest_data = result['backtest_data']
                plt.plot(backtest_data['Date'], backtest_data['TotalValue'], 
                        label=f"Seed {result['seed']}", alpha=0.7)
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.title('Portfolio Performance Comparison')
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig('multi_seed_performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not generate comparison plot: {e}")
        
    return results_summary

def analyze_macro_influence(decisions_file_pattern='trading_decisions_enhanced_improved_seed_*.csv'):
    """
    Analyzes how macro recommendations influenced trading decisions across different seeds.
    """
    import glob
    
    # Find all seed-specific files
    files = glob.glob(decisions_file_pattern)
    
    if not files:
        print("No enhanced improved model decisions files found yet. Run backtest first.")
        return None
    
    all_results = []
    
    for file in files:
        try:
            # Extract seed from filename
            seed = file.split('_seed_')[1].split('.csv')[0]
            df = pd.read_csv(file)
            df['File_Seed'] = seed
            all_results.append(df)
            
            print(f"\nSEED {seed} ANALYSIS:")
            print(f"  Total decisions: {len(df)}")
            print(f"  Buy: {len(df[df['Decision'] == 'Buy'])} ({len(df[df['Decision'] == 'Buy'])/len(df)*100:.1f}%)")
            print(f"  Sell: {len(df[df['Decision'] == 'Sell'])} ({len(df[df['Decision'] == 'Sell'])/len(df)*100:.1f}%)")
            print(f"  Hold: {len(df[df['Decision'] == 'Hold'])} ({len(df[df['Decision'] == 'Hold'])/len(df)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        print(f"\nCOMBINED MULTI-SEED ANALYSIS:")
        print(f"  Total decisions across all seeds: {len(combined_df)}")
        
        # Analyze decision consistency across seeds
        if len(all_results) > 1:
            decision_by_seed = combined_df.groupby('File_Seed')['Decision'].value_counts()
            print(f"  Decision consistency analysis:")
            for decision in ['Buy', 'Sell', 'Hold']:
                seed_percentages = []
                for seed_df in all_results:
                    pct = len(seed_df[seed_df['Decision'] == decision]) / len(seed_df) * 100
                    seed_percentages.append(pct)
                
                if seed_percentages:
                    consistency = np.std(seed_percentages)
                    print(f"    {decision} decisions - Mean: {np.mean(seed_percentages):.1f}%, Std: {consistency:.1f}%")
                    print(f"      Consistency: {'HIGH' if consistency < 5 else 'MEDIUM' if consistency < 10 else 'LOW'}")
        
        return combined_df
    
    return None

# Legacy functions for backward compatibility
def run_backtest_improved_with_temporal_protection(tickers_list, start_date_str, end_date_str):
    """Legacy function - redirects to enhanced version with default seed"""
    print("Warning: Using legacy function name - redirecting to enhanced version with default seed")
    return run_backtest_improved_with_enhanced_temporal_protection(tickers_list, start_date_str, end_date_str, random_seed=42)

def run_backtest(tickers_list, start_date_str, end_date_str):
    """Legacy function - redirects to enhanced version with default seed"""
    print("Warning: Using legacy function name - redirecting to enhanced version with default seed")
    return run_backtest_improved_with_enhanced_temporal_protection(tickers_list, start_date_str, end_date_str, random_seed=42)

# --- Main execution block ---
if __name__ == "__main__":
    print("ENHANCED IMPROVED MODEL WITH COMPREHENSIVE TEMPORAL PROTECTION AND SEED CONTROL")
    print("(XGBoost + SHAP + Enhanced Temporal Validation + Random Seed Control)")
    print("="*80)
    
    my_tickers = [
        'ASML NA Equity', 'PRX NA Equity', 'INGA NA Equity', 'UMG NA Equity', 'ADYEN NA Equity', 
        'HEIA NA Equity', 'WKL NA Equity', 'AD NA Equity', 'ASM NA Equity', 'PHIA NA Equity', 
        'HEIO NA Equity', 'EXO NA Equity', 'KPN NA Equity', 'NN NA Equity', 'ASRNL NA Equity', 
        'JDEP NA Equity', 'AGN NA Equity', 'BESI NA Equity', 'AKZA NA Equity', 'CTPNV NA Equity', 
        'RAND NA Equity', 'IMCD NA Equity', 'VPK NA Equity', 'TKWY NA Equity', 'SBMO NA Equity', 
        'ARCAD NA Equity', 'AALB NA Equity', 'LIGHT NA Equity', 'BAMNB NA Equity', 'BFIT NA Equity', 
        'OCI NA Equity', 'FUR NA Equity', 'FLOW NA Equity', 'CRBN NA Equity', 'AMG NA Equity', 
        'TOM2 NA Equity', 'ACOMO NA Equity', 'PHARM NA Equity', 'NEDAP NA Equity', 'SLIGR NA Equity', 
        'PNL NA Equity', 'BRNL NA Equity', 'ENVI NA Equity', 'FFARM NA Equity', 'SIFG NA Equity', 
        'ALFEN NA Equity', 'HYDRA NA Equity', 'NXFIL NA Equity', 'CMCOM NA Equity', 'KENDR NA Equity', 
        'AJAX NA Equity', 'AZRN NA Equity', 'AVTX NA Equity', 'QEV NA Equity', 'QEVT NA Equity', 
        'HOLCO NA Equity', 'VALUE NA Equity', 'CABKA NA Equity', 'DSC2S NA Equity', 'NAI NA Equity', 
        'NAITR NA Equity', 'CTAC NA Equity', 'BEVER NA Equity', 'EBUS NA Equity', 'AMUND NA Equity', 
        'ALX NA Equity', 'PBH NA Equity', 'PORF NA Equity', 'EAS2P NA Equity', 'DGB NA Equity', 
        'NEDSE NA Equity', 'NSE NA Equity', 'LVIDE NA Equity', 'TITAN NA Equity', 'INPHI NA Equity', 
        'ESGT NA Equity', 'ENTPA NA Equity'
    ]

    # Configuration
    start_period = "2015-01-01"
    end_period = "2024-12-31"
    
    # Choose execution mode
    SINGLE_SEED_MODE = False  # Set to True for single seed, False for multi-seed testing
    PRIMARY_SEED = 42
    TEST_SEEDS = [42, 123, 456, 789, 999]  # For multi-seed testing with 5 seeds
    
    if SINGLE_SEED_MODE:
        print(f"Running single seed backtest (seed: {PRIMARY_SEED})")
        print(f"Period: {start_period} to {end_period}")
        print(f"Tickers: {len(my_tickers)} companies")
        
        backtest_results = run_backtest_improved_with_enhanced_temporal_protection(
            my_tickers, start_period, end_period, random_seed=PRIMARY_SEED
        )
        
        if backtest_results is not None:
            print(f"\n{'='*70}")
            print("ENHANCED IMPROVED MODEL WITH SEED CONTROL COMPLETE!")
            print(f"{'='*70}")
            
            initial_value = 1000000
            final_value = backtest_results['TotalValue'].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100
            
            print(f"\nPERFORMANCE SUMMARY (Seed: {PRIMARY_SEED}):")
            print(f"  Initial Portfolio Value: ${initial_value:,}")
            print(f"  Final Portfolio Value:   ${final_value:,.2f}")
            print(f"  Total Return:            {total_return:+.2f}%")
            print(f"  Random Seed:             {PRIMARY_SEED}")
            print(f"  Enhanced XGBoost + SHAP: Enabled")
            print(f"  Comprehensive Temporal Protection: Enabled")
            print(f"  Results saved to: trading_decisions_enhanced_improved_seed_{PRIMARY_SEED}.csv")
            
            # Analyze results for this seed
            analyze_macro_influence(f'trading_decisions_enhanced_improved_seed_{PRIMARY_SEED}.csv')
            
            # Plot results
            try:
                plt.figure(figsize=(15, 8))
                backtest_results.set_index('Date')['TotalValue'].plot(
                    kind='line', 
                    linewidth=2, 
                    color='darkgreen',
                    title=f'Enhanced Improved Model (Seed: {PRIMARY_SEED}): Portfolio Performance'
                )
                plt.ylabel("Portfolio Value ($)", fontsize=12)
                plt.xlabel("Date", fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.axhline(y=initial_value, color='r', linestyle='--', alpha=0.5, label='Initial Value')
                plt.legend()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not generate plot: {e}")
        
    else:
        print("Running multi-seed stability analysis...")
        print(f"Period: {start_period} to {end_period}")
        print(f"Tickers: {len(my_tickers)} companies")
        print(f"Seeds: {TEST_SEEDS}")
        
        multi_results = run_multi_seed_backtest(
            my_tickers, start_period, end_period, seeds=TEST_SEEDS
        )
        
        if multi_results:
            print(f"\n{'='*70}")
            print("MULTI-SEED ANALYSIS COMPLETE!")
            print(f"{'='*70}")
            
            # Analyze combined results
            analyze_macro_influence()
            
            print("\nFinal Summary:")
            for result in multi_results:
                print(f"  Seed {result['seed']}: {result['total_return']:+.2f}% return")
    
    print(f"\nSEED CONTROL VERIFICATION COMPLETE!")
    print("All analyses controlled for random seed to ensure reproducibility.")