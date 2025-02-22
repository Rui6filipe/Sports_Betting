# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:40:41 2024

@author: ruira
"""
#%%
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import optuna
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from functools import partial
from scipy.stats import gaussian_kde 
from scipy.stats import norm
from properscoring import crps_gaussian
from numba import njit

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%%

''' Bayesian Neural Network Class'''

#%%
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma_w=0.5, prior_sigma_b=0.5,
                     weight_mu_range=(-0.1, 0.1), weight_rho_range=(-5.5, -2.5)):
        super().__init__()
        
        # Weight parameters
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(*weight_mu_range))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(*weight_rho_range))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(*weight_mu_range))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(*weight_rho_range))
        
        # Prior distribution parameters
        self.prior_sigma_w = prior_sigma_w
        self.prior_sigma_b = prior_sigma_b
        
        # Initialize parameters
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias_mu)

    def forward(self, x):
        
        # Compute weight and bias standard deviations
        weight_sigma = torch.log1p(torch.exp(self.weight_rho)) + 1e-8
        bias_sigma = torch.log1p(torch.exp(self.bias_rho)) + 1e-8
        
        # Reparameterization trick for sampling weights and biases
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu) 
        
        # Sample weights and biases
        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """
        Compute KL divergence between posterior and prior
        """
        weight_sigma = torch.log1p(torch.exp(self.weight_rho)) + 1e-8
        bias_sigma = torch.log1p(torch.exp(self.bias_rho)) + 1e-8
        
        # KL divergence for weights
        weight_kl = 0.5 * torch.sum(
            2 * torch.log(self.prior_sigma_w / weight_sigma) + 
            (weight_sigma**2 + (self.weight_mu - 0)**2) / (self.prior_sigma_w**2) - 1)

        # KL divergence for biases
        bias_kl = 0.5 * torch.sum(
            2 * torch.log(self.prior_sigma_b / bias_sigma) + 
            (bias_sigma**2 + (self.bias_mu - 0)**2) / (self.prior_sigma_b**2) - 1)

        return weight_kl + bias_kl


class BayesianNN(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()
        
        # Extract hyperparameters from params dictionary
        hidden_size1 = params.get('units_layer1', 128)
        hidden_size2 = params.get('units_layer2', 64)
        hidden_size3 = params.get('units_layer3', 32)
        hidden_size4 = params.get('units_layer4', 16)

        self.deterministic_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0)
        )
        # Bayesian layers
        self.bayesian_layer1 = BayesianLinear(input_size, hidden_size1)
        self.bayesian_layer2 = BayesianLinear(hidden_size1, hidden_size2)
        self.bayesian_layer3 = BayesianLinear(hidden_size2, hidden_size3)
        self.bayesian_layer4 = BayesianLinear(hidden_size3, hidden_size4)
        self.bayesian_output = BayesianLinear(hidden_size4, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.bayesian_layer1(x), negative_slope=0.01)
        x = F.leaky_relu(self.bayesian_layer2(x), negative_slope=0.01)
        x = F.leaky_relu(self.bayesian_layer3(x), negative_slope=0.01)
        x = F.leaky_relu(self.bayesian_layer4(x), negative_slope=0.01)
        x = self.bayesian_output(x)
        return x
    

    def sample_elbo(self, input, target, loss_fn, sample_weights, kl_weight=0, num_samples=100, log = False):
        """
        Compute the Evidence Lower Bound (ELBO) loss for a Bayesian Neural Network.
        """      
        sample_weights = sample_weights.view(-1, 1)
    
        predictions = []
        
        for _ in range(num_samples):
            output = self(input)  
            predictions.append(output)
        
        predictions = torch.stack(predictions)
        pred_mean = predictions.mean(dim=0)
        pred_var = predictions.var(dim=0) + 1e-8

        #nll = 0.5 * torch.log(2 * torch.pi * pred_var) + ((target - pred_mean) ** 2) / (2 * pred_var)
        nll = crps_gaussian_torch(target, pred_mean, torch.sqrt(pred_var))
 
        prediction_loss = (nll * sample_weights).mean()
        
        kl_div = (
            self.bayesian_layer1.kl_loss() + 
            self.bayesian_layer2.kl_loss() +
            self.bayesian_layer3.kl_loss() +
            self.bayesian_layer4.kl_loss() +
            self.bayesian_output.kl_loss())
        
        total_loss = prediction_loss + kl_weight * kl_div
        
        #if log:
        #    print("\nPrediction Loss :", prediction_loss)
        #    print("KL Loss :", kl_weight * kl_div)
        #    print("Components :", kl_div)
    
        return total_loss
  
    
def crps_gaussian_torch(target, mean, std):
    """Computes CRPS for Gaussian distribution in pure PyTorch."""
    standardized = (target - mean) / std  # (y - μ) / σ
    cdf = 0.5 * (1 + torch.erf(standardized / torch.sqrt(torch.tensor(2.0))))  # Φ(x)
    pdf = (1 / torch.sqrt(torch.tensor(2.0) * torch.pi)) * torch.exp(-0.5 * standardized ** 2)  # φ(x)

    crps = std * (standardized * (2 * cdf - 1) + 2 * pdf - 1 / torch.sqrt(torch.tensor(torch.pi)))
    return crps


def betting_profit_loss_torch(actual, pred_mean, pred_std, odds=1.66, threshold1=6.5, threshold2=7.5):
    """
    Loss function that optimizes for Kelly Criterion-based expected profit.
    Penalizes bad bets and rewards profitable ones.
    """
    #thresholds = torch.tensor(6.5, dtype=torch.float32, device=actual.device)
    thresholds = torch.full_like(actual, 6.5, dtype=torch.float32)
    #thresholds = torch.where(actual > 7.5, torch.tensor(threshold2, device=actual.device), torch.tensor(threshold1, device=actual.device))
    
    normal_dist = torch.distributions.Normal(pred_mean, pred_std)
    prob_under = normal_dist.cdf(thresholds)

    prob_over = 1 - prob_under  
    best_prob = torch.maximum(prob_over, prob_under)
    implied_prob = 1 / odds
    payout = odds - 1

    edge = best_prob - implied_prob  
    #kelly_fraction = torch.clamp(edge / payout, 0)
    kelly_fraction = torch.sigmoid(edge / payout) 

    bet_won = torch.sigmoid((prob_over - prob_under) * (actual - thresholds))
    #bet_won = (prob_over > prob_under) == (actual > thresholds)

    profit = kelly_fraction * (bet_won * payout - (1 - bet_won))

    return (1 - profit)

#%%

''' DATA PART'''

#%%
def load_and_preprocess_data(csv_path):
    """
    Load and preprocess hockey game data from CSV
    """
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    df['game_date'] = pd.to_datetime(df['game_date'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    
    df['average_points_per_game_1'] = (3 * df['win_percentage_1'] + 1 * df['draw_percentage_1'])
    df = df.drop(columns=['win_percentage_1', 'draw_percentage_1', 'loss_percentage_1'])
    df['average_points_per_game_2'] = (3 * df['win_percentage_2'] + 1 * df['draw_percentage_2'])
    df = df.drop(columns=['win_percentage_2', 'draw_percentage_2', 'loss_percentage_2'])

    # Split the DataFrame into training/testing and future games
    future_games_df = df.tail(7).reset_index(drop=True)
    train_test_df = df.iloc[:-7].reset_index(drop=True)
    
    # Encode categorical variables in both datasets
    categorical_columns = ['team1', 'team2']
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        label_encoders[col].fit(train_test_df[col])
        train_test_df[f'{col}_encoded'] = label_encoders[col].transform(train_test_df[col])
        future_games_df[f'{col}_encoded'] = label_encoders[col].transform(future_games_df[col])

    features = [col for col in train_test_df.columns]
    
    # Remove any unwanted columns
    features_to_exclude = ['team1', 'team2', 'home_team', 'game_date', 'goals_scored',
                           'avg_goals_scored_home_1', 'avg_goals_scored_home_2', 
                           'avg_goals_scored_away_1', 'avg_goals_scored_away_2',
                           'avg_goals_conceded_home_1', 'avg_goals_conceded_home_2',
                           'avg_goals_conceded_away_1', 'avg_goals_conceded_away_2',
                           'team_sos_avg_points_1', 'team_sos_avg_goal_difference_1', 
                           'team_sos_avg_points_2', 'team_sos_avg_goal_difference_2',
                           'avg_fouls_favor_1', 'avg_fouls_against_1', 'avg_fouls_favor_2',
                           'avg_fouls_against_2', 'avg_cards_favor_1', 'avg_cards_against_1',
                           'avg_cards_favor_2', 'avg_cards_against_2', 'rest_1', 'rest_2',
                           'goals_scored_trend_1', 'goals_conceded_trend_1', 'goals_scored_trend_2',
                           'goals_conceded_trend_2', 'ref1_avg_fouls', 'ref1_avg_cards', 'ref2_avg_fouls',
                           'ref2_avg_cards','penalties_tried_1', 'penalties_scored_1', 
                           'penalties_tried_against_1','penalties_conceded_1', 'penalties_tried_2', 
                           'penalties_scored_2', 'penalties_tried_against_2', 'penalties_conceded_2',
                           'dfh_tried_1', 'dfh_scored_1', 
                           'dfh_tried_against_1','dfh_conceded_1', 'dfh_tried_2', 
                           'dfh_scored_2', 'dfh_tried_against_2', 'dfh_conceded_2']
    
    features_use = [f for f in features if f not in features_to_exclude]

    return train_test_df, future_games_df, label_encoders, features_use


def print_dataset_info(df):
    """
    Print useful information about the dataset
    """
    print("Dataset Information")
    print(f"Number of games: {len(df)}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique teams: {df['team1'].nunique()}")


#%%

''' FEATURES PART'''

#%%
def analyze_feature_correlations(df, features, threshold):
    """
    Find highly correlated features
    Returns pairs of features with correlation above threshold
    """
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                high_corr_pairs.append((
                    features[i], 
                    features[j], 
                    corr_matrix.iloc[i,j]))
    
    return high_corr_pairs


def select_features(df, features, target, threshold):
    """
    Select features based on correlation and importance
    """
    # Find highly correlated features
    high_corr_pairs = analyze_feature_correlations(df, features, threshold)
    
    # Train a model
    model = XGBRegressor(objective='count:poisson') 
    model.fit(df[features], df[target])
    
    # Feature importance with XGBoost 
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # For highly correlated pairs, keep the one with higher importance
    features_to_drop = set()
    for feat1, feat2, corr in high_corr_pairs:
        imp1 = importance_df[importance_df['feature'] == feat1]['importance'].values[0]
        imp2 = importance_df[importance_df['feature'] == feat2]['importance'].values[0]
        if imp1 < imp2:
            features_to_drop.add(feat1)
        else:
            features_to_drop.add(feat2)
    
    selected_features = [f for f in features if f not in features_to_drop]
    return selected_features, importance_df, high_corr_pairs


def plot_feature_importance(importance_df, top_n=20):
    """
    Plot feature importance from XGBoost
    """
    plt.figure(figsize=(12, 8))
    sns.barplot(data=importance_df.head(top_n), 
                x='importance', y='feature')
    plt.title('Top Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, features):
    """
    Plot correlation heatmap
    """
    plt.figure(figsize=(15, 12))
    
    correlation_matrix = df[features].corr()
    
    sns.heatmap(
        correlation_matrix,
        annot=True,          # Show numbers
        cmap='RdBu',        # Red-Blue diverging colormap
        center=0,           # Center the colormap at 0
        fmt='.2f',         # Show 2 decimal places
        square=True,       # Make cells square
        annot_kws={'size': 7},  # Make numbers a bit smaller
        cbar_kws={'shrink': .8}  # Slightly smaller colorbar
    ) 
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title('Correlation Heatmap', pad=20)
    plt.tight_layout()

    return plt


#%%

''' TRAINING PREPARATION PART'''

#%%
def train_test_split(df, n_ignore, n_test): 
    """
    Split data based on rounds and train-test splits where:
    - Training data consists of all previous seasons and rounds from the current season.
    - Test data consists of the next round in the current season.
    """
    seasons = ["2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617"]
    
    # Create a mask to filter out the first n games of each season
    remove_indices = []
    for season in seasons:
        season_df = df[df["team1"].str.endswith(season)]
        season_indices = season_df.head(n_ignore).index  # First n games for this season
        remove_indices.extend(season_indices)
    df = df.drop(remove_indices)
    
    train_test_splits = []

    current_season_df = df[df['team1'].str.endswith("2425")]
    last_21_games = current_season_df.tail(n_test)
    test_game_dates = last_21_games["game_date"].unique()
    
    for current_date in test_game_dates:
        test_data = df[df["game_date"] == current_date]
        train_data = df[df["game_date"] < current_date]
        X_train = train_data
        y_train = train_data["goals_scored"]
        X_test = test_data
        y_test = test_data["goals_scored"]
        
        train_test_splits.append((X_train, y_train, X_test, y_test))

    return train_test_splits


def validation_split(df, n_ignore, n_test, n_prev_season): 
    """
    Split data based on rounds and train-test splits where:
    - Training data consists of all previous seasons and rounds from the current season.
    - Test data consists of the next round in the current season.
    """
    seasons = ["2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718", "1617"]
    
    # Create a mask to filter out the first n games of each season
    remove_indices = []
    for season in seasons:
        season_df = df[df["team1"].str.endswith(season)]
        season_indices = season_df.head(n_ignore).index  # First n games for this season
        remove_indices.extend(season_indices)
    df = df.drop(remove_indices)
    
    validation_splits = []

    previous_season_df = df[df['team1'].str.endswith("2324")]
    last_30_previous_season = previous_season_df.tail(n_prev_season)
    current_season_df = df[df['team1'].str.endswith("2425")]
    current_season_excl_test = current_season_df.iloc[:-n_test]
    validation_games = pd.concat([last_30_previous_season, current_season_excl_test])
    unique_dates = validation_games["game_date"].unique()
    
    for current_date in unique_dates:
        val_data = df[df["game_date"] == current_date]
        train_data = df[df["game_date"] < current_date]
        X_train = train_data
        y_train = train_data["goals_scored"]
        X_val = val_data
        y_val = val_data["goals_scored"]
        
        validation_splits.append((X_train, y_train, X_val, y_val))
    
    return validation_splits


def create_time_weights(dates): 
    """
    Create weights that give more importance to recent seasons
    """
    '''season_ranges = [ 
        (datetime(2024, 9, 1), datetime(2025, 9, 1), 10),
        (datetime(2023, 9, 1), datetime(2024, 9, 1), 2),
        (datetime(2022, 9, 1), datetime(2023, 9, 1), 2),
        (datetime(2021, 9, 1), datetime(2022, 9, 1), 2),
        (datetime(2020, 9, 1), datetime(2021, 9, 1), 1),
        (datetime(2019, 9, 1), datetime(2020, 9, 1), 1),
        (datetime(2018, 9, 1), datetime(2019, 9, 1), 1),
        (datetime(2017, 9, 1), datetime(2018, 9, 1), 1),
        (datetime(2016, 9, 1), datetime(2017, 9, 1), 1)]'''
    
    season_ranges = [ 
        (datetime(2024, 9, 1), datetime(2025, 9, 1), 1),
        (datetime(2023, 9, 1), datetime(2024, 9, 1), 1),
        (datetime(2022, 9, 1), datetime(2023, 9, 1), 1),
        (datetime(2021, 9, 1), datetime(2022, 9, 1), 1),
        (datetime(2020, 9, 1), datetime(2021, 9, 1), 1),
        (datetime(2019, 9, 1), datetime(2020, 9, 1), 1),
        (datetime(2018, 9, 1), datetime(2019, 9, 1), 1),
        (datetime(2017, 9, 1), datetime(2018, 9, 1), 1),
        (datetime(2016, 9, 1), datetime(2017, 9, 1), 1)]
    
    weights = []
    # Check which season the game belongs to and assign the corresponding weight
    for date in dates:
        for start_date, end_date, season_weight in season_ranges:
            if start_date <= date < end_date:
                weights.append(season_weight)
                break
    
    weights = np.array(weights)
    return weights / np.max(weights)  


def betting_profit_loss(actual, dist, odds=1.66, threshold1=6.5, threshold2=7.5):
    """
    Loss function that optimizes for Kelly Criterion-based expected profit.
    Penalizes bad bets and rewards profitable ones.
    """
    pred_mean = np.mean(dist) 
    pred_std = np.std(dist) + 1e-8
    
    thresholds = np.where(actual > 7.5, threshold2, threshold1)
    
    prob_under = norm.cdf(thresholds, loc=pred_mean, scale=pred_std)
    #prob_under = 1 / (1 + np.exp(-1.702 * (thresholds - pred_mean) / pred_std))
    prob_over = 1 - prob_under

    best_prob = np.maximum(prob_over, prob_under)

    implied_prob = 1 / odds
    payout = odds - 1  

    edge = best_prob - implied_prob
    kelly_fraction = np.maximum(edge / payout, 0) 

    bet_won = (prob_over > prob_under) == (actual > thresholds)

    profit = kelly_fraction * (bet_won * payout - (1 - bet_won))

    return (1 - profit)   



def get_baseline_score(train_test_splits, features):
    """
    Get baseline performance using Random Forest
    """
    mae = []
    for i, (X_training, y_train, X_test, y_test) in enumerate(train_test_splits, start=1):
        
        #weights = create_time_weights(X_training['game_date'])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_training[features])
        X_test = scaler.transform(X_test[features])
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_predictions = rf_model.predict(X_test)
        baseline_mae = np.mean(np.abs(rf_predictions - y_test))
        mae.extend(np.abs(rf_predictions - y_test))
        print(f"MAE for split {i} is: {baseline_mae}")
        
    print("Test mean MAE: ", np.mean(mae))


def find_learning_rate(model, X_train, y_train, sample_weights, criterion, start_lr=1e-6, end_lr=1, num_iterations=1000):
    """ Implementation of learning rate finder without batch processing """
    lrs = []
    losses = []
    log_lrs = np.linspace(np.log10(start_lr), np.log10(end_lr), num_iterations)
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    model.train()
    
    for i, lr in enumerate(np.power(10, log_lrs)):
        try:
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad()
            
            loss = model.sample_elbo(X_train, y_train, criterion, sample_weights)
            loss.backward()
            optimizer.step()
            
            lrs.append(lr)
            losses.append(loss.item())
            
            if i % 100 == 0:
                print(f"\nIteration {i}")
                print(f"Learning rate: {lr:.2e}")
                print(f"Loss: {loss.item():.6f}")
                
        except Exception as e:
            print(f"Error at learning rate {lr}: {str(e)}")
            break
    
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.ylim(0, 200)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.grid(True)
    plt.show()
    
    smooth_losses = np.convolve(losses, np.ones(50)/50, mode='same')
    gradient = np.gradient(smooth_losses)
    
    min_lr_index = np.searchsorted(lrs, 2e-4)   
    max_lr_index = np.searchsorted(lrs, 2e-2)  
    relative_optimal_idx = np.argmin(gradient[min_lr_index:max_lr_index])  
    optimal_idx = min_lr_index + relative_optimal_idx
        
    optimal_lr = lrs[optimal_idx]
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, smooth_losses)
    plt.ylim(0, 200)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs Loss')
    plt.grid(True)
    plt.show()
    
    return optimal_lr


def setup_lr(X_training, y_train, features, initial_params):
    """
    Setup model and find maximum learning rate using FastAI's lr_find.
    """
    weights = create_time_weights(X_training['game_date'])   
    sample_weights = torch.FloatTensor(weights)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_training[features])
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
     
    test_model = BayesianNN(X_train.shape[1], initial_params)   
    criterion = nn.MSELoss(reduction='none')
    optimal_lr = find_learning_rate(test_model, X_train_tensor, y_train_tensor, sample_weights, criterion)
    print(f"Maximum learning rate found: {optimal_lr:.6f}")
    
    return optimal_lr



def train(bnn_model, optimizer, criterion, X_train, y_train, sample_weights, sample_weights_val,  
          X_val, y_val, num_epochs, optimal_lr, kl_weight=0, num_samples=100, trial=0, patience=100, 
          mode="early_stop"):
    """
    Train the Bayesian Neural Network with early stopping.
    
    Returns:
    - bnn_model: The trained model, restored to the best state.
    """
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    train_losses = []
    train_losses_nll = [] 
    val_losses_nll = [] 
    val_losses = []
    train_losses_mae = [] 
    val_losses_mae = []
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=optimal_lr,
        epochs=num_epochs,
        steps_per_epoch=1,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,        
        final_div_factor=100) 
    
    for epoch in range(num_epochs):
        bnn_model.train()
        optimizer.zero_grad()
        
        if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch == 299:
            train_loss = bnn_model.sample_elbo(
                X_train, 
                y_train, 
                criterion, 
                sample_weights,
                kl_weight, log=(mode=="early_stop"))
            
        else: 
            train_loss = bnn_model.sample_elbo(
                X_train, 
                y_train, 
                criterion, 
                sample_weights,
                kl_weight)
            
        train_loss.backward()
        optimizer.step()
        scheduler.step()
        
        bnn_model.eval()
        with torch.no_grad():
            train_distributions = predict_distribution_bnn(bnn_model, X_train, num_samples)
            train_pred_mean = train_distributions.mean(axis=1)  
            train_pred_uncertainty = train_distributions.std(axis=1) 

            train_log_likelihood = [
                betting_profit_loss(actual, dist) 
                for actual, dist in zip(y_train.numpy().flatten(), train_distributions)]
            
            train_nll = np.mean(train_log_likelihood)
            train_mae = np.mean(np.abs(train_pred_mean - y_train.numpy().flatten()))  

        train_losses.append(train_loss.item())
        train_losses_nll.append(train_nll)
        train_losses_mae.append(train_mae)
        residuals_train = train_pred_mean - y_train.numpy().flatten()
        uncertainties_train = train_pred_uncertainty
        
        if mode == "early_stop":
            bnn_model.eval()
            with torch.no_grad():
                val_distributions = predict_distribution_bnn(bnn_model, X_val, num_samples)
                val_pred_mean = val_distributions.mean(axis=1)
                val_pred_uncertainty = val_distributions.std(axis=1)
  
                if epoch == 49 or epoch == 99 or epoch == 149 or epoch == 199 or epoch == 249 or epoch == 299:
                    val_loss = bnn_model.sample_elbo(X_val, y_val, criterion, 
                                                   sample_weights_val, kl_weight, log = True)
                else: 
                    val_loss = bnn_model.sample_elbo(X_val, y_val, criterion, 
                                                   sample_weights_val, kl_weight)
                
                val_log_likelihood = [
                    betting_profit_loss(actual, dist) 
                    for actual, dist in zip(y_val.numpy().flatten(), val_distributions)]
                
                val_nll = np.mean(val_log_likelihood)
                val_mae = np.mean(np.abs(val_pred_mean - y_val.numpy().flatten()))

            val_losses.append(val_loss.item())
            val_losses_nll.append(val_nll)
            val_losses_mae.append(val_mae)
            residuals_val = val_pred_mean - y_val.numpy().flatten()
            uncertainties_val = val_pred_uncertainty

            #if (epoch + 1) % 50 == 0:
                #print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss = {train_loss.item():.4f} | Train NLL: {train_nll:.4f} | Train MAE: {train_mae:.4f} | Val Loss = {val_loss.item():.4f} | Val NLL: {val_nll:.4f} | Val MAE: {val_mae:.4f}")
    
            if val_nll < best_val_loss and epoch>60:
                best_val_loss = val_nll
                epochs_without_improvement = 0
                best_model_state = copy.deepcopy(bnn_model.state_dict())
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Stopping early at epoch {epoch + 1}. No improvement for {patience} epochs.")
                break

    if mode == "early_stop" and best_model_state is not None:
        bnn_model.load_state_dict(best_model_state)

    # --- Plot Results ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.ylim(0, 20)
    plt.grid(True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_nll, label='Train NLL', color='blue')
    plt.plot(val_losses_nll, label='Val NLL', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Negative Log-Likelihood (NLL)")
    plt.title(f"Training vs Validation NLL, trial {trial-1}")
    plt.ylim(0, 2)
    plt.legend()
    plt.grid(True)
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses_mae, label='Train MAE', color='blue')
    plt.plot(val_losses_mae, label='Val MAE', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title(f"Training vs Validation MAE, trial {trial-1}")
    plt.ylim(0, 10)
    plt.legend()
    plt.grid(True)
    
    if mode == "early_stop":
  
        plt.figure(figsize=(8, 5))
        plt.scatter(abs(residuals_val), uncertainties_val, alpha=0.5, label="Validation")
        plt.scatter(abs(residuals_train), uncertainties_train, alpha=0.5, label="Train")
        plt.xlabel("Residuals (Prediction Error)")
        plt.ylabel("Uncertainty (Standard Deviation)")
        plt.title("Uncertainty vs. Residuals")
        plt.axhline(y=np.mean(uncertainties_train), color='r', linestyle='--', label="Mean Uncertainty")
        plt.axvline(x=0, color='k', linestyle='--')
        plt.legend()
        plt.grid(True)
 
    plt.tight_layout()
    plt.show()

    return bnn_model

#%%

''' VALIDATION PART'''

#%%
def objective(optimal_lr, trial):
    """
    Objective function for Bayesian optimization using Optuna.
    """
    '''learning_rate = trial.suggest_uniform("learning_rate", optimal_lr/2, optimal_lr*3/2) 
    units_layer1 = trial.suggest_categorical("units_layer1", [128, 256])  
    units_layer2 = trial.suggest_categorical("units_layer2", [64, 128]) 
    units_layer3 = trial.suggest_categorical("units_layer3", [32, 64])
    units_layer4 = trial.suggest_categorical("units_layer4", [16, 32])
    epochs = trial.suggest_categorical("epochs", [150, 200, 250])
    kl_weight = trial.suggest_loguniform("kl_weight", 0.00001, 0.0001)'''
    
    learning_rate = trial.suggest_uniform("learning_rate", optimal_lr-optimal_lr/5, optimal_lr+optimal_lr/5) 
    epochs = 150
    kl_weight = trial.suggest_uniform("kl_weight", 1e-6, 4e-6)

    params = {
        'learning_rate': learning_rate,
        'units_layer1': 128,
        'units_layer2': 64,
        'units_layer3': 16,
        'units_layer4': 8,
        'epochs': 150,
        'kl_weight': kl_weight}

    all_log_likelihoods = []
    all_maes = []

    for X_training, y_train, X_validation, y_val in validation_splits:
        weights = create_time_weights(X_training['game_date'])
        sample_weights = torch.FloatTensor(weights)
        weights_val = create_time_weights(X_validation['game_date'])
        sample_weights_val = torch.FloatTensor(weights_val)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_training[features])
        X_val = scaler.transform(X_validation[features])
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val.values).unsqueeze(1)

        bnn_model = BayesianNN(X_train.shape[1], params)
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(bnn_model.parameters(), lr=learning_rate)
        
        trained_model = train(
            bnn_model, optimizer, criterion, 
            X_train, y_train, sample_weights, 
            sample_weights_val, X_val, y_val, 
            epochs, optimal_lr, kl_weight,
            trial = trial.number + 1)

        trained_model.eval()
        with torch.no_grad():
            distributions = predict_distribution_bnn(trained_model, X_val)
            y_pred = distributions.mean(axis=0)  
        
        for actual, dist in zip(y_val.detach().numpy().flatten(), distributions): 
            log_likelihood = betting_profit_loss(actual, dist)
            all_log_likelihoods.extend(np.atleast_1d(log_likelihood))
            all_maes.extend(np.abs(actual - y_pred))
 
    mean_log_likelihood = np.mean(all_log_likelihoods)
    mean_mae = np.mean(all_maes)
    
    print(f"\nTrial {trial.number}")
    print(f"Mean Absolute Error (MAE): {mean_mae:.4f}")
    
    return mean_log_likelihood  



def tune_hyperparameters(validation_splits, features, n_trials, optimal_lr):
    """
    Run Bayesian Optimization for hyperparameter tuning.
    """
    print("Start hyperparameter tuning")
    
    objective_with_param = partial(objective, optimal_lr)
    
    study = optuna.create_study(direction="minimize")  
    study.optimize(objective_with_param, n_trials=n_trials)

    print("\nBest hyperparameters found:")
    print(f"Trial Number: {study.best_trial.number}")
    print(f"Hyperparameters: {study.best_params}")
    print(f"Best Objective Value: {study.best_value}")

    return study.best_params

#%%

''' TRAINING, TESTING AND PREDICTING PART'''

#%%
def train_and_evaluate(train_test_splits, features, best_params, thresholds, game_to_see, test_round_index, optimal_lr):
    """
    Train and evaluate the model on each train-test split, reporting the performance.
    
    :param train_test_splits: List of train-test splits
    :param best_params: Hyperparameters for RandomizedSearchCV
    :return: List of model evaluation results (e.g., RMSE)
    """
    results = [] 
    results_NLL = []        
    true = []   
    pred = []  
    distr = []
    i = 0
    probs = []
    
    for X_training, y_train, X_test, y_test in train_test_splits:

        weights = create_time_weights(X_training['game_date'])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_training[features])
        X_test_scaled = scaler.transform(X_test[features])
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        bnn_model = BayesianNN(X_train.shape[1], best_params)
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(bnn_model.parameters(), lr=optimal_lr)
        
        trained_model = train(
            bnn_model, optimizer, criterion, 
            X_train_tensor, y_train_tensor, weights_tensor,  
            weights_tensor, X_test_tensor, y_test_tensor, 
            best_params["epochs"], optimal_lr, best_params["kl_weight"],
            mode = "test")
            
        trained_model.eval()
        with torch.no_grad():
            distributions = predict_distribution_bnn(trained_model, X_test_tensor)
            y_pred = distributions.mean(axis=1)  
        
        log_likelihood_scores = []
        for actual, dist in zip(y_test, distributions): 
            log_likelihood = betting_profit_loss(actual, dist)
            log_likelihood_scores.append(log_likelihood)

        log_likelihood_score = np.mean(log_likelihood_scores)
        mae = mean_absolute_error(y_test_tensor.numpy().flatten(), y_pred)
        results.append(mae)
        results_NLL.extend(np.atleast_1d(log_likelihood_scores))
        
        distributions = np.array(distributions)
        distribuiton, probss = display_predictions(X_test, y_pred, distributions, thresholds, "test", y_test, i)
        print(f"\nTESTE MAE: {mae:.4f}, Negative Log-Likelihood: {log_likelihood_score:.4f}")

        probs.append(probss)
        true.append(y_test)  
        pred.append(y_pred)
        distr.append(distribuiton)
        i += 1
    
    true_flat = np.concatenate(true)
    pred_flat = np.concatenate(pred)
    dist_flat = np.concatenate(distr) 
    
    plot_prediction_vs_actual(np.array(true_flat), np.array(pred_flat))
    plot_test_results(np.array(true_flat), np.array(pred_flat), np.array(dist_flat))
    plot_prediction_distribution(distr, true, test_round_index, game_to_see)
    
    print(probs)
    
    return trained_model, scaler, np.mean(results), np.mean(results_NLL)


def predict_distribution_bnn(model, X, n_simulations=100):
    """
    Generate probability distribution of predictions using BNN
    """
    samples = np.array([model(X).detach().numpy().flatten() for _ in range(n_simulations)])
    distributions = samples.T
    
    return distributions


def predict(df_future, model, X_new, thresholds):
    """
    Make predictions on new data.
    """
    model.eval()
    with torch.no_grad():
        X_new = torch.tensor(X_new, dtype=torch.float32)
        y_pred = model(X_new).flatten()
    
    print(y_pred)
    distributions = predict_distribution_bnn(model, X_new, 200)
    distributions = np.array(distributions)
    display_predictions(df_future, y_pred, distributions, thresholds, "new")



def calculate_betting_metrics(probabilities, odds, idx):
    """
    Calculate Expected Value (EV) and the fraction to bet using the Kelly Criterion.
    """
    EVs = []
    kelly_fractions = []
        
    implied_probability = 1 / odds[idx]
    payout = odds[idx] - 1
    my_probability = probabilities[0]
    EV = (my_probability * payout) - (1 - my_probability)
    EVs.append(EV)
    edge = my_probability - implied_probability
    kelly_fraction = edge / payout
    kelly_fractions.append(kelly_fraction)
    
    if probabilities[1] == 0:
        print("\nUnder")
    elif probabilities[1] == 1:
        print("\nOver")
    
    print(f"My Probability: {my_probability * 100:.2f}%")
    print(f"Implied Probability: {implied_probability * 100:.2f}%")
    print(f"Expected Value (EV): {EV * 100:.2f}%")
    print(f"Kelly Criterion Fraction to Bet: {kelly_fraction * 100:.2f}%")
    print("\n")
    
    return EVs, kelly_fractions, my_probability

#%%

''' Display Part'''

#%%
def display_predictions(df, predictions, distributions, thresholds, mode="test", results=None, i=None):
    """
    Display predictions and optionally compare with actual results for test or new games.

    :param data: DataFrame containing game data (test or new games).
    :param model: Trained model to make predictions.
    :param thresholds: List of thresholds for over/under probabilities.
    :param mode: "test" for test games, "new" for new games.
    :param results: Actual outcomes (only for "test" mode).
    :param i: Index for train-test split (only for "test" mode).
    """
    print(f"\nPREDICTIONS FOR {'TEST' if mode == 'test' else 'NEW'} GAMES {f'{i}' if i else ''}")
    print("=" * 80)
    
    idx = 0
    for _,game in df.iterrows():
        print(f"\nGame {idx + 1}: {game['team1']} vs {game['team2']}")
        print(f"Date: {game['game_date'].strftime('%Y-%m-%d')}")
        print("-" * 40)
        
        print(f"Predicted Average Goals: {predictions[idx]:.2f}")
        if mode == "test" and results is not None:
            actual_result = results.iloc[idx]
            print(f"Actual Result: {actual_result} goals")
        
        print("\nOver/Under Probabilities:")
        for threshold in thresholds:
            over_prob = np.mean(distributions[idx] > threshold)
            print(f"Over {threshold}: {over_prob:.1%}")
            print(f"Under {threshold}: {1 - over_prob:.1%}")

        print("\nProbability Distribution:")
        unique_goals = np.unique(distributions[idx])
        for goals in unique_goals:
            prob = np.mean(distributions[idx] == goals)
            if prob >= 0.05:  # Show probabilities >= 5%
                print(f"{goals} goals: {prob:.1%}")
       
        over_prob = np.mean(distributions[idx] > assumed_tresholds[idx])
        highest_prob = max(1 - over_prob, over_prob)
        ov_und = 1 if over_prob > (1 - over_prob) else 0
        probabilities = [highest_prob, ov_und]
        _, _, probs = calculate_betting_metrics(probabilities, odds, idx)
        print("\n" + "=" * 80)
        idx = (idx+1)%7
    
    return distributions, probs


def plot_prediction_vs_actual(y_true, y_pred):
    """
    Plot predicted vs actual values
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', label='Perfect Prediction')
    
    plt.title('Predicted vs Actual Goals')
    plt.xlabel('Actual Goals')
    plt.ylabel('Predicted Goals')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_test_results(target, predictions, distributions, thresholds=[5.5, 6.5, 7.5]):
    """
    Create a standalone visualization of over/under performance.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Compute over/under performance results
    over_under_results = []
    for threshold in thresholds:
        actual_over = (target > threshold).mean() 
        pred_over = (predictions > threshold).mean()
        over_under_results.append({
            'Threshold': threshold,
            'Actual Over %': actual_over * 100,
            'Predicted Over %': pred_over * 100
        })

    results_df = pd.DataFrame(over_under_results)

    # Create bar plot
    x = range(len(thresholds))
    width = 0.35
    plt.figure(figsize=(10, 7))
    plt.bar(x, results_df['Actual Over %'], width, label='Actual', color='blue', alpha=0.6)
    plt.bar([i + width for i in x], results_df['Predicted Over %'], width, 
            label='Predicted', color='red', alpha=0.6)
    plt.xticks([i + width / 2 for i in x], [f'Over {t}' for t in thresholds])
    plt.title('Over/Under Performance')
    plt.ylabel('Percentage of Games')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
    
def plot_prediction_distribution(distributions, actual, test_round_index, game_index):
    """
    Plot predicted distribution for a specific game
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of predicted distribution
    sns.histplot(distributions[test_round_index][game_index], 
                stat='probability',
                discrete=True)
    
    # Add vertical line for actual value if provided
    actual_value = actual[test_round_index].iloc[game_index]
    if actual_value is not None:
        plt.axvline(x=actual_value,  
                   color='r', 
                   linestyle='--', 
                   label='Actual')
        plt.legend()
    
    plt.title('Predicted Goals Distribution for Game {}'.format(game_index + 1))
    plt.xlabel('Number of Goals')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show()
   
#%%
# Load and preprocess the data
df, df_future, encoders, features = load_and_preprocess_data('final_game_features.csv')

# Print information about dataset
print_dataset_info(df)

# Remove excess features
correlation_threshold = 0.85
selected_features, importance_df, high_corr_pairs = select_features(
                df, features, 'goals_scored', correlation_threshold) 

plot_feature_importance(importance_df)
plot_correlation_heatmap(df, selected_features)

# Splits and baseline
n_ignore = 63 #42
n_test = 63-21 # 63
n_prev_season = 35
n_iterations = 8
validation_splits = validation_split(df, n_ignore, n_test, n_prev_season)
train_test_splits = train_test_split(df, n_ignore, n_test)
#get_baseline_score(train_test_splits, selected_features)

# Tune hyperparameters
initial_params = {
    'units_layer1': 256,
    'units_layer2': 128,
    'units_layer3': 64,
    'units_layer4': 32}
optimal_lr = 0.01
#setup_lr(train_test_splits[0][0], train_test_splits[0][1], selected_features, initial_params)

#best_params = tune_hyperparameters(validation_splits, selected_features, n_iterations, optimal_lr)
best_params = {'learning_rate': 0.01, 'units_layer1': 128, 'units_layer2': 64,
               'units_layer3': 16, 'units_layer4': 8, 'epochs': 150, 'kl_weight': 1.5e-6}
print("\nBest parameters:", best_params)


# Train and evaluate model
thresholds = [5.5, 6.5, 7.5, 8.5] 
game_to_see = 0
test_round_index = 2
odds = [1.66, 1.66, 1.66, 1.66, 1.66, 1.66, 1.66]
assumed_tresholds = [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5]
model, scaler, results, results_NLL = train_and_evaluate(train_test_splits, 
                    selected_features, best_params, thresholds, 
                    game_to_see, test_round_index, optimal_lr)
print("\nAverage MAE:", results)
print("\nAverage NLL:", results_NLL)
print("\n" + "=" * 80)

new_game_data = scaler.transform(df_future[selected_features])
predict(df_future, model, new_game_data, thresholds)
