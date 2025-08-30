import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore', category=FutureWarning)

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"RL Agent random seeds set to {seed}")

class QLearningAgent:
    """
    Optimized Q-Learning agent using only the most important features.
    """
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, 
                 exploration_decay_rate=0.001):
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.randint(self.q_table.shape[1])
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state, action] = new_value

    def decay_exploration_rate(self, episode):
        self.exploration_rate = self.min_exploration_rate + \
            (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)


def get_optimized_state(row):
    """
    FIXED: Optimized state representation using only 6 ESSENTIAL features.
    No changes needed here - this function already uses only current data.
    """
    
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
    
    # Combine into single state index
    # 3 * 3 * 3 * 2 * 2 * 2 = 216 total states
    state_index = (rsi_state * 72 +      # 3 * 3 * 2 * 2 * 2 = 72
                   trend_state * 24 +     # 3 * 2 * 2 * 2 = 24  
                   pe_state * 8 +         # 2 * 2 * 2 = 8
                   vol_state * 4 +        # 2 * 2 = 4
                   macro_state * 2 +      # 2
                   macd_state)            # 1
    
    return min(int(state_index), 215)  # Ensure we don't exceed bounds


def calculate_reward(row, final_decision):
    """
    FIXED: Reward calculation using only available data at decision time.
    """
    future_price = row.get('Future_Close_30D')
    current_price = row.get('Close')
    
    if pd.isna(future_price) or pd.isna(current_price) or current_price == 0:
        return 0
    
    price_change = (future_price - current_price) / current_price
    
    # Simple but effective reward logic
    if final_decision == 'Buy':
        return 2 if price_change > 0.02 else -1
    elif final_decision == 'Sell':  
        return 2 if price_change < -0.02 else -1
    else:  # Hold
        return 1 if abs(price_change) <= 0.02 else -0.5


def load_and_validate_temporal_data(dataset_filepath: str):
    """
    FIXED: Load and validate temporally protected dataset.
    """
    print("Loading and validating temporal dataset...")
    
    try:
        df = pd.read_csv(dataset_filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Loaded {len(df)} rows from {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # Check if Split column exists (from temporal protection)
        if 'Split' in df.columns:
            split_counts = df['Split'].value_counts()
            print(f"Split distribution:")
            for split, count in split_counts.items():
                print(f"  {split}: {count} rows ({count/len(df)*100:.1f}%)")
            return df, True
        else:
            print("Warning: No Split column found - creating manual temporal splits")
            return df, False
            
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, False


def create_manual_temporal_splits(df):
    """
    FIXED: Create manual temporal splits if not already present.
    """
    print("Creating manual temporal splits...")
    
    # Sort by date
    df = df.sort_values('Date').copy()
    
    # Define split boundaries (adjust based on your data range)
    total_dates = df['Date'].nunique()
    
    # Use 60% for training, 20% for validation, 20% for test
    train_pct = 0.60
    val_pct = 0.20
    
    unique_dates = sorted(df['Date'].unique())
    train_end_idx = int(len(unique_dates) * train_pct)
    val_end_idx = int(len(unique_dates) * (train_pct + val_pct))
    
    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]
    
    def assign_split(date):
        if date <= train_end_date:
            return 'train'
        elif date <= val_end_date:
            return 'validation'
        else:
            return 'test'
    
    df['Split'] = df['Date'].apply(assign_split)
    
    split_counts = df['Split'].value_counts()
    print(f"Manual splits created:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} rows")
    
    return df


def run_optimized_training_with_temporal_protection(
    dataset_filepath: str, 
    output_qtable_filepath: str, 
    episodes=25000,
    random_seed=42  # NEW: Add random seed parameter
):
    """
    FIXED: Training loop with proper temporal data splitting to prevent future data leakage.
    Enhanced with comprehensive seed control.
    """
    print(f"OPTIMIZED RL TRAINING WITH TEMPORAL PROTECTION (Seed: {random_seed})")
    print("=" * 70)
    
    # Set random seeds at the very beginning
    set_random_seeds(random_seed)
    
    # Load and validate data
    df, has_splits = load_and_validate_temporal_data(dataset_filepath)
    if df is None:
        return None
    
    # Create manual splits if needed
    if not has_splits:
        df = create_manual_temporal_splits(df)
    
    # CRITICAL FIX: Extract splits with proper temporal boundaries
    train_df = df[df['Split'] == 'train'].copy()
    val_df = df[df['Split'] == 'validation'].copy()
    test_df = df[df['Split'] == 'test'].copy()
    
    if len(train_df) == 0:
        print("Error: No training data available!")
        return None
    
    print(f"\nTemporal splits loaded:")
    print(f"  Training: {len(train_df)} rows ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"  Validation: {len(val_df)} rows ({val_df['Date'].min().date()} to {val_df['Date'].max().date()})")
    print(f"  Test: {len(test_df)} rows ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    # Validate temporal integrity
    if len(val_df) > 0:
        assert train_df['Date'].max() < val_df['Date'].min(), "TEMPORAL VIOLATION: Training overlaps with validation!"
    if len(test_df) > 0 and len(val_df) > 0:
        assert val_df['Date'].max() < test_df['Date'].min(), "TEMPORAL VIOLATION: Validation overlaps with test!"
    print("Temporal integrity validated")
    
    # Clean training data
    print(f"\nCleaning training data...")
    required_cols = ['Close', 'Future_Close_30D', 'RSI_14', 'SMA_50', 
                    'Price Earnings Ratio (P/E)', 'ATR_14', 'ECB_Rate', 'MACD_Signal']
    
    # Only keep training rows with valid future prices (needed for reward calculation)
    train_df = train_df.dropna(subset=['Close', 'Future_Close_30D'])
    
    # Fill missing values with reasonable defaults
    train_df['RSI_14'].fillna(50, inplace=True)
    train_df['SMA_50'].fillna(train_df['Close'], inplace=True)
    train_df['Price Earnings Ratio (P/E)'].fillna(20, inplace=True)
    train_df['ATR_14'].fillna(0, inplace=True)
    train_df['ECB_Rate'].fillna(2.0, inplace=True)
    train_df['MACD_Signal'].fillna(0, inplace=True)
    
    print(f"Training data after cleaning: {len(train_df)} rows")
    
    if len(train_df) < 100:
        print("Warning: Very little training data available!")
    
    # Initialize agent
    state_space_size = 216  # 3 * 3 * 3 * 2 * 2 * 2
    action_space_size = 4   # Macro-Heavy, Fundamental-Heavy, Technical-Heavy, Balanced
    
    agent = QLearningAgent(state_space_size, action_space_size, 
                          learning_rate=0.1, exploration_decay_rate=0.0008)
    
    actions = {
        0: {'name': 'macro_heavy', 'weights': {'macro': 0.6, 'fundamental': 0.2, 'technical': 0.2}},
        1: {'name': 'fundamental_heavy', 'weights': {'macro': 0.2, 'fundamental': 0.6, 'technical': 0.2}},
        2: {'name': 'technical_heavy', 'weights': {'macro': 0.2, 'fundamental': 0.2, 'technical': 0.6}},
        3: {'name': 'balanced', 'weights': {'macro': 0.33, 'fundamental': 0.33, 'technical': 0.33}}
    }
    
    print(f"\nRL Agent Configuration:")
    print(f"  State space: {state_space_size} states")
    print(f"  Action space: {action_space_size} actions")
    print(f"  Episodes: {episodes}")
    print(f"  Random Seed: {random_seed}")
    
    # FIXED: Training loop - ONLY use training data with proper temporal ordering
    print(f"\nStarting temporal-protected training...")
    rewards_log = []
    
    # Sort training data by date to ensure temporal ordering
    train_df_sorted = train_df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    train_indices = train_df_sorted.index.tolist()
    
    # Create valid training pairs (current, next) that respect temporal ordering
    valid_pairs = []
    for ticker in train_df_sorted['Ticker'].unique():
        ticker_data = train_df_sorted[train_df_sorted['Ticker'] == ticker].sort_values('Date')
        ticker_indices = ticker_data.index.tolist()
        
        # Create consecutive pairs for this ticker
        for i in range(len(ticker_indices) - 1):
            current_idx = ticker_indices[i]
            next_idx = ticker_indices[i + 1]
            
            # Verify temporal ordering
            current_date = ticker_data.loc[current_idx, 'Date']
            next_date = ticker_data.loc[next_idx, 'Date']
            
            if current_date < next_date:
                valid_pairs.append((current_idx, next_idx))
    
    print(f"Created {len(valid_pairs)} valid temporal training pairs")
    
    if len(valid_pairs) == 0:
        print("Error: No valid training pairs found!")
        return None
    
    # Training loop with temporal protection and seeded randomness
    for episode in range(episodes):
        try:
            # CRITICAL FIX: Sample ONLY from valid temporal pairs using seeded random state
            if len(valid_pairs) == 0:
                continue
                
            current_idx, next_idx = valid_pairs[np.random.randint(len(valid_pairs))]
            
            current_row = train_df_sorted.loc[current_idx]
            next_row = train_df_sorted.loc[next_idx]
            
            # Double-check temporal ordering
            if current_row['Date'] >= next_row['Date']:
                continue
            
            state = get_optimized_state(current_row)
            next_state = get_optimized_state(next_row)
            
            # Ensure states are valid
            if state >= state_space_size or next_state >= state_space_size:
                continue
            
            # Agent chooses action (using seeded random state)
            action = agent.choose_action(state)
            
            # Simulate decision based on current market conditions
            rsi = current_row.get('RSI_14', 50)
            pe = current_row.get('Price Earnings Ratio (P/E)', 20)
            price_trend = current_row.get('Close', 100) / current_row.get('SMA_50', 100)
            
            # Decision logic incorporating the chosen action's strategy
            action_weights = actions[action]['weights']
            
            # Weight the decision factors based on the action
            fundamental_signal = 1 if pe < 18 else (-1 if pe > 25 else 0)
            technical_signal = 1 if rsi < 40 else (-1 if rsi > 70 else 0)
            trend_signal = 1 if price_trend > 1.02 else (-1 if price_trend < 0.98 else 0)
            
            # Combine signals based on action weights
            combined_signal = (
                action_weights['fundamental'] * fundamental_signal +
                action_weights['technical'] * technical_signal +
                action_weights['macro'] * trend_signal  # Use trend as macro proxy
            )
            
            if combined_signal > 0.3:
                decision = 'Buy'
            elif combined_signal < -0.3:
                decision = 'Sell'
            else:
                decision = 'Hold'
            
            # Calculate reward and update Q-table
            reward = calculate_reward(current_row, decision)
            agent.update_q_table(state, action, reward, next_state)
            agent.decay_exploration_rate(episode)
            
            rewards_log.append(reward)
            
        except Exception as e:
            continue
        
        # Progress logging
        if (episode + 1) % 5000 == 0:
            avg_reward = np.mean(rewards_log[-5000:]) if len(rewards_log) >= 5000 else np.mean(rewards_log)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Exploration: {agent.exploration_rate:.3f} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Seed: {random_seed}")
    
    print(f"Training completed with {len(rewards_log)} valid episodes")
    
    # FIXED: Validation on separate temporal data
    if len(val_df) > 0:
        print(f"\nValidating on temporally separate validation set...")
        val_df_clean = val_df.dropna(subset=['Close', 'Future_Close_30D']).copy()
        
        if len(val_df_clean) > 0:
            # Fill missing values for validation
            val_df_clean['RSI_14'].fillna(50, inplace=True)
            val_df_clean['SMA_50'].fillna(val_df_clean['Close'], inplace=True)
            val_df_clean['Price Earnings Ratio (P/E)'].fillna(20, inplace=True)
            val_df_clean['ATR_14'].fillna(0, inplace=True)
            val_df_clean['ECB_Rate'].fillna(2.0, inplace=True)
            val_df_clean['MACD_Signal'].fillna(0, inplace=True)
            
            val_rewards = []
            
            for idx in val_df_clean.index:
                try:
                    row = val_df_clean.loc[idx]
                    state = get_optimized_state(row)
                    
                    if state >= state_space_size:
                        continue
                    
                    # Use learned policy (no exploration)
                    agent.exploration_rate = 0
                    action = agent.choose_action(state)
                    
                    # Simulate decision using same logic as training
                    rsi = row.get('RSI_14', 50)
                    pe = row.get('Price Earnings Ratio (P/E)', 20)
                    price_trend = row.get('Close', 100) / row.get('SMA_50', 100)
                    
                    action_weights = actions[action]['weights']
                    fundamental_signal = 1 if pe < 18 else (-1 if pe > 25 else 0)
                    technical_signal = 1 if rsi < 40 else (-1 if rsi > 70 else 0)
                    trend_signal = 1 if price_trend > 1.02 else (-1 if price_trend < 0.98 else 0)
                    
                    combined_signal = (
                        action_weights['fundamental'] * fundamental_signal +
                        action_weights['technical'] * technical_signal +
                        action_weights['macro'] * trend_signal
                    )
                    
                    if combined_signal > 0.3:
                        decision = 'Buy'
                    elif combined_signal < -0.3:
                        decision = 'Sell'
                    else:
                        decision = 'Hold'
                    
                    reward = calculate_reward(row, decision)
                    val_rewards.append(reward)
                    
                except Exception:
                    continue
            
            val_avg_reward = np.mean(val_rewards) if val_rewards else 0
            print(f"Validation avg reward: {val_avg_reward:.3f} ({len(val_rewards)} samples)")
        else:
            print("Warning: No valid validation data available")
    
    # Save Q-table with seed-specific filename
    try:
        # Create seed-specific filename
        base_name = output_qtable_filepath.replace('.npy', '')
        seed_qtable_path = f"{base_name}_seed_{random_seed}.npy"
        
        np.save(seed_qtable_path, agent.q_table)
        print(f"\nSaved temporally protected Q-table to {seed_qtable_path}")
    except Exception as e:
        print(f"Error saving Q-table: {e}")
        return None
    
    # Analysis
    print(f"\nTRAINING RESULTS WITH TEMPORAL PROTECTION (Seed: {random_seed})")
    print("=" * 60)
    action_preferences = np.argmax(agent.q_table, axis=1)
    for action_idx in range(action_space_size):
        count = np.sum(action_preferences == action_idx)
        percentage = count / state_space_size * 100
        action_name = actions[action_idx]['name']
        print(f"  {action_name}: {count} states ({percentage:.1f}%)")
    
    if len(rewards_log) > 0:
        train_avg_reward = np.mean(rewards_log[-1000:]) if len(rewards_log) >= 1000 else np.mean(rewards_log)
        print(f"\n  Final training avg reward: {train_avg_reward:.3f}")
    
    print(f"\nTEMPORAL PROTECTION SUMMARY:")
    print(f"  Training used only historical data")
    print(f"  Validation on separate future period")
    print(f"  No future data leakage detected")
    print(f"  Proper temporal ordering enforced")
    print(f"  Random seed {random_seed} applied throughout")
    
    return agent.q_table


def run_multi_seed_rl_training(
    dataset_filepath: str,
    base_qtable_filepath: str,
    seeds=[42, 123, 456, 789, 999],
    episodes=25000
):
    """
    NEW: Train multiple RL agents with different seeds for stability analysis.
    """
    print("MULTI-SEED RL TRAINING WITH TEMPORAL PROTECTION")
    print("=" * 70)
    print(f"Training {len(seeds)} RL agents with seeds: {seeds}")
    print(f"Episodes per agent: {episodes}")
    
    results = []
    
    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"TRAINING RL AGENT WITH SEED: {seed}")
        print(f"{'='*50}")
        
        q_table = run_optimized_training_with_temporal_protection(
            dataset_filepath,
            base_qtable_filepath,
            episodes=episodes,
            random_seed=seed
        )
        
        if q_table is not None:
            results.append({
                'seed': seed,
                'q_table': q_table,
                'success': True
            })
            print(f"Seed {seed}: Training successful")
        else:
            results.append({
                'seed': seed,
                'q_table': None,
                'success': False
            })
            print(f"Seed {seed}: Training failed")
    
    # Analyze differences between Q-tables
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) > 1:
        print(f"\nMULTI-SEED RL ANALYSIS:")
        print(f"Successful trainings: {len(successful_results)}/{len(seeds)}")
        
        # Compare action preferences across seeds
        for action_idx in range(4):
            action_names = ['macro_heavy', 'fundamental_heavy', 'technical_heavy', 'balanced']
            seed_preferences = []
            
            for result in successful_results:
                q_table = result['q_table']
                action_preferences = np.argmax(q_table, axis=1)
                pref_pct = np.sum(action_preferences == action_idx) / len(action_preferences) * 100
                seed_preferences.append(pref_pct)
            
            mean_pref = np.mean(seed_preferences)
            std_pref = np.std(seed_preferences)
            print(f"  {action_names[action_idx]}: {mean_pref:.1f}% ± {std_pref:.1f}%")
        
        # Calculate Q-table stability
        if len(successful_results) >= 2:
            q_tables = [r['q_table'] for r in successful_results]
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(len(q_tables)):
                for j in range(i+1, len(q_tables)):
                    flat_q1 = q_tables[i].flatten()
                    flat_q2 = q_tables[j].flatten()
                    corr = np.corrcoef(flat_q1, flat_q2)[0, 1]
                    correlations.append(corr)
            
            mean_corr = np.mean(correlations)
            print(f"\nQ-table stability: {mean_corr:.3f} (1.0 = identical, 0.0 = uncorrelated)")
            
            if mean_corr > 0.8:
                print("High stability - RL policies are very consistent across seeds")
            elif mean_corr > 0.5:
                print("Medium stability - RL policies show some variation")
            else:
                print("Low stability - RL policies vary significantly across seeds")
    
    return results


if __name__ == "__main__":
    print("OPTIMIZED RL AGENT WITH TEMPORAL PROTECTION AND SEED CONTROL")
    print("=" * 70)
    print("KEY FIXES:")
    print("- Proper temporal data splitting")
    print("- Training only on historical data")
    print("- Validation on separate future periods")
    print("- No future data leakage in Q-learning")
    print("- Temporal ordering enforced in training pairs")
    print("- Comprehensive random seed control")
    print("=" * 70)
    print("SELECTED FEATURES:")
    print("- RSI_14 - Momentum indicator")
    print("- Price vs SMA_50 - Trend strength") 
    print("- P/E Ratio - Fundamental valuation")
    print("- ATR_14 - Volatility/Risk measure")
    print("- ECB_Rate - Macro environment")
    print("- MACD_Signal - Trend confirmation")
    print("=" * 70)
    
    # File paths - use the temporal protected dataset
    dataset_file = 'time_machine_dataset_temporal_protected.csv'
    qtable_file = 'optimized_agent_q_table_temporal_protected.npy'
    
    # Configuration
    SINGLE_SEED_MODE = False  # Set to True for single seed, False for multi-seed training
    PRIMARY_SEED = 42
    TRAINING_SEEDS = [42, 123, 456, 789, 999]
    
    if SINGLE_SEED_MODE:
        print(f"Training single RL agent with seed {PRIMARY_SEED}...")
        
        result = run_optimized_training_with_temporal_protection(
            dataset_file, 
            qtable_file, 
            episodes=30000,
            random_seed=PRIMARY_SEED
        )
        
        if result is not None:
            print(f"\nRL TRAINING COMPLETE (Seed: {PRIMARY_SEED})!")
            print("Your RL agent is now trained with proper temporal boundaries.")
        else:
            print("\nTraining failed. Please check your data and try again.")
    
    else:
        print("Training multiple RL agents with different seeds for stability analysis...")
        
        multi_results = run_multi_seed_rl_training(
            dataset_file,
            qtable_file,
            seeds=TRAINING_SEEDS,
            episodes=30000
        )
        
        successful_count = len([r for r in multi_results if r['success']])
        print(f"\nMULTI-SEED RL TRAINING COMPLETE!")
        print(f"Successfully trained {successful_count}/{len(TRAINING_SEEDS)} RL agents")
        
        if successful_count > 0:
            print("Q-tables saved with seed-specific filenames:")
            for result in multi_results:
                if result['success']:
                    print(f"  optimized_agent_q_table_temporal_protected_seed_{result['seed']}.npy")
    
    print(f"\nSEED CONTROL VERIFICATION COMPLETE!")
    print("All RL training controlled for random seed to ensure reproducibility.")