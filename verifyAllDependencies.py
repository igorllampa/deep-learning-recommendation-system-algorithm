import torch
import pandas as pd
import numpy as np
import datetime
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

print("✅ All dependencies are installed correctly!")