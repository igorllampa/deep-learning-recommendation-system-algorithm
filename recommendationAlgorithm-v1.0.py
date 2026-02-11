import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def log_execution(
    algorithm_name_and_version,
    start_time,
    end_time,
    epochs,
    embedding,    
    database_test_name,    
    validationParams
):    
    log_entry = (
    f"AlgorithmNameAndVersion: {algorithm_name_and_version}\n"
    f"DateTimeBegin: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"DateTimeEnd: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    f"Epochs: {epochs}\n"
    f"Embedding: {embedding}\n"
    f"DataBaseTestName: {database_test_name}\n"
    f"Threshold: {validationParams['threshold']}\n"
    f"Precision: {validationParams['precision']}\n"
    f"Recall: {validationParams['recall']}\n"
    f"F1-Score: {validationParams['f1score']}\n"
    f"Accuracy: {validationParams['accuracy']}\n"
    f"RMSE: {validationParams['rmse']}\n"
    f"MAE: {validationParams['mae']}\n"
    f"{'-'*40}\n\n"
)    

    with open("logs/"+algorithm_name_and_version+"-log.txt", "a") as file: 
        file.write(log_entry)
    

def load_data(pathFile):
    
    # Load ratings
    ratingsFile = pathFile+'/ratings.txt'
    ratings = pd.read_csv(
        ratingsFile, sep='\t', header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine='python'
    )
    
    # Load movies
    moviesFile = pathFile+'/movies.txt'
    movies = pd.read_csv(moviesFile, sep='|', encoding='ISO-8859-1', header=None, 
                         names=["movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"])

    # Load users
    usersFile = pathFile+'/users.txt'
    users = pd.read_csv(
        usersFile, sep='|', header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        engine='python'
    )

    print(ratings)
    print(movies)
    print(users)

    return ratings, movies, users

def get_top_n_recommendations(model, user_id, user_features, item_features, movies, n=10):
    """
    Generates the top N recommendations for a specific user.

    :param model: Trained model
    :param user_id: ID of the user for whom recommendations are to be generated
    :param user_features: Dictionary containing user characteristics
    :param item_features: numpy.ndarray containing item features
    :param movies: DataFrame containing movie information
    :param n: Number of recommendations to display
    """
    model.eval()
    recommendations = []

    print(f"User {user_id} Features: {user_features[user_id]}")
    print(f"Item Features Sample: {item_features[:5]}")
    
    # Retrieves the user's feature vector
    user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
    
    for item_id in range(len(item_features)):
        item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)
        score = model(user_feat, item_feat).item()
        movie_title = movies.loc[movies['movie_id'] == item_id, 'movie_title'].values
        movie_title = movie_title[0] if len(movie_title) > 0 else "Unknown"
        recommendations.append((item_id, movie_title, score))
    
    # Sort the recommendations by score in descending order
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    # Display the top N results
    print("Top", n, "recomendações para o usuário", user_id)
    print("="*80)
    for i, (movie_id, title, score) in enumerate(recommendations[:n], 1):
        print(f"{i:05d}. ID: {movie_id:05d} | {title[:58]:<58} | Score: {score:.4f}")
    print("="*80)
    
    return recommendations[:n]


def get_top_N_collaborative(model, user_id, item_features, user_features, item_ids, movies_df, users_df, N=100, device='cpu'):
    """
    Retrieves the top **N** recommendations using collaborative filtering.

    **Adds**:
    - Parameter `N` to define how many items to display
    - Displays user information and the names of the recommended movies
    """
    
    model.eval()
    user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0).to(device)
    
    scores = []
    
    with torch.no_grad():
        for item_id in item_ids:
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0).to(device)
            score = model(user_feat, item_feat).item()
            scores.append((item_id, score))
    
    # Sort the recommendations by score and select the top **N** results
    top_N = sorted(scores, key=lambda x: x[1], reverse=True)[:N]
    
    print("\n **User Information for Recommendation:**")
    user_info = users_df[users_df['user_id'] == user_id]
    print(user_info.to_string(index=False))  # Display without Pandas indexes

    print(f"\n **Top {N} Movie Recommendations (Collaborative Filtering):**")
    for i, (item_id, score) in enumerate(top_N, 1):
        movie_info = movies_df[movies_df['movie_id'] == item_id]
        movie_title = movie_info["movie_title"].values[0] if not movie_info.empty else "Unknown"                
        movie_title = movie_title[:55].ljust(55)
        print(f"{i:05d}. ID: {item_id:05d} | {movie_title} | Score: {score:.4f}")

    return [item[0] for item in top_N]  


def get_top_N_content_based(model, user_id, item_features, user_features, occupation_features, title_features, year_features, genre_features, item_ids, movies_df, users_df, N=100, device='cpu'):
    """
    Retrieves the top **N** recommendations using a content-based model.

    **Adds**:
    - Parameter `N` to define how many items to display
    - Displays user information and the names of the recommended movies
    """
    
    model.eval()
    user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0).to(device)
    occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0).to(device)
    
    scores = []
    
    with torch.no_grad():
        for item_id in item_ids:
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0).to(device)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0).to(device)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0).to(device)
            
            score = model(user_feat, occ_feat, title_feat, year_feat, genre_feat).item()
            scores.append((item_id, score))
    
    # Sort the items by score and retrieve the top **N** recommendations
    top_N = sorted(scores, key=lambda x: x[1], reverse=True)[:N]
    
    print("\n **User Information Used for Recommendation:**")
    user_info = users_df[users_df['user_id'] == user_id]
    print(user_info.to_string(index=False))  # Display without Pandas indeex

    print(f"\n **Top {N} Movie Recommendations (Content-Based Filtering):**")
    for i, (item_id, score) in enumerate(top_N, 1):
        movie_info = movies_df[movies_df['movie_id'] == item_id]
        movie_title = movie_info["movie_title"].values[0] if not movie_info.empty else "Unknown"        
        movie_title = movie_title[:55].ljust(55)
        print(f"{i:05d}. ID: {item_id:05d} | {movie_title} | Score: {score:.4f}")

    return [item[0] for item in top_N] 


def preprocess_data(ratings, movies, users):

    user_to_index = {user_id: idx for idx, user_id in enumerate(ratings['user_id'].unique())}
    item_to_index = {item_id: idx for idx, item_id in enumerate(ratings['item_id'].unique())}

    ratings['user_id'] = ratings['user_id'].map(user_to_index)
    ratings['item_id'] = ratings['item_id'].map(item_to_index)

    # Fix division by zero issue in normalization
    if ratings['rating'].max() - ratings['rating'].min() == 0:
        ratings['rating'] = 0.5  # Assign a fixed value if normalization is impossible
    else:
        ratings['rating'] = (ratings['rating'] - ratings['rating'].min()) / (ratings['rating'].max() - ratings['rating'].min())

    users['user_id'] = users['user_id'].map(user_to_index)
    user_features = users[['age', 'gender']].copy()
    user_features['gender'] = user_features['gender'].map({'M': 0, 'F': 1})
    user_features = user_features.fillna(0).values.astype(np.float32)  # Replace NaN with 0
    occupation_features = pd.get_dummies(users['occupation']).fillna(0).values.astype(np.float32)

    movies['movie_id'] = movies['movie_id'].map(item_to_index)
    movies.drop(columns=['video_release_date', 'IMDb_URL'], inplace=True)  # Remove non-numeric columns

    # Extract year safely, filling NaN with 0
    year_features = movies['release_date'].str.extract(r'(\d{4})').astype(float).fillna(0).values.reshape(-1, 1)

    # Extract last 19 columns as genre, ensuring no NaN values
    genre_features = movies.iloc[:, 5:].fillna(0).values.astype(np.float32)

    # Convert movie titles to numerical features using one-hot encoding
    title_features = pd.get_dummies(movies['movie_title']).values.astype(np.float32)

    item_features = np.concatenate([title_features, year_features, genre_features], axis=1)  

    return ratings, len(user_to_index), len(item_to_index), user_features, occupation_features, title_features, year_features, genre_features, item_features
    
class ContentBasedModel(nn.Module):
    def __init__(self, user_dim, occupation_dim, title_dim, year_dim, genre_dim, embedding_dim):
        super(ContentBasedModel, self).__init__()
        
        print('# Sub-network for Age and Gender feature processing')
        # Sub-network for Age and Gender feature processing
        self.user_branch = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 54),
            nn.ReLU(),
            nn.Linear(54, 32),
            nn.ReLU()
        )
        
        print('# Sub-network for Occupation feature processing')
        # Sub-network for Occupation feature processing
        self.occupation_branch = nn.Sequential(
            nn.Linear(occupation_dim, 11),
            nn.ReLU()
        )
        
        print('# Sub-network for Title feature processing')
        # Sub-network for Title feature processing
        self.title_branch = nn.Sequential(
            nn.Linear(title_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        print('# Sub-network for Year feature processing')
        # Sub-network for Year feature processing
        self.year_branch = nn.Sequential(
            nn.Linear(year_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        print('# Sub-network for Genre feature processing')
        # Sub-network for Genre feature processing
        self.genre_branch = nn.Sequential(
            nn.Linear(genre_dim, 9),
            nn.ReLU()
        )
        
        print('# Final layer that combines all feature representations')
        # Final layer that combines all feature representations
        self.final_layers = nn.Sequential(
            nn.Linear(32 + 11 + 256 + 32 + 9, 32),  # Concatenating the outputs from all sub-networks
            nn.ReLU(),
            nn.Linear(32, 1)  # Single final output for recommendation
        )
    
    def forward(self, user_input, occupation_input, title_input, year_input, genre_input):
        user_out = self.user_branch(user_input)
        occupation_out = self.occupation_branch(occupation_input)
        title_out = self.title_branch(title_input)
        year_out = self.year_branch(year_input)
        genre_out = self.genre_branch(genre_input)

        # Ensure tensors have the same batch size before concatenation
        if user_out.dim() == 1:
            user_out = user_out.unsqueeze(0)
        if occupation_out.dim() == 1:
            occupation_out = occupation_out.unsqueeze(0)
        if title_out.dim() == 1:
            title_out = title_out.unsqueeze(0)
        if year_out.dim() == 1:
            year_out = year_out.unsqueeze(0)
        if genre_out.dim() == 1:
            genre_out = genre_out.unsqueeze(0)

        combined = torch.cat((user_out, occupation_out, title_out, year_out, genre_out), dim=1)
        return self.final_layers(combined)


class CollaborativeFilteringDeepLearning(nn.Module):
    def __init__(self, user_dim, item_dim, embedding_dim):
        super(CollaborativeFilteringDeepLearning, self).__init__()

        # MLP for extracting user features
        self.user_branch = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # MLP for extracting item features
        self.item_branch = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

        # SDAE-FM (AutoEncoder + Matrix Factorization)
        self.sdae_fm = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        #  DNN (Deep Neural Network with LayerNorm instead of BatchNorm)
        self.dnn = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.LayerNorm(128),  # Use LayerNorm instead
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # LayerNorm does not depend on batch size
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        #  Metric Learning (Convolucional)        
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1)  # Reduce from 16 to 8 filters
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1) # Reduce from 32 to 16 filters        

        self.fc_metric = nn.Linear(3840, 1)  # Change from 960 to 3840


    def forward(self, user_input, item_input):
        # Feature extraction for users and items
        user_features = self.user_branch(user_input)
        item_features = self.item_branch(item_input)

        # Concatenating the latent vectors
        combined_features = torch.cat((user_features, item_features), dim=1)

        # Passing through the 3 sub-modules
        sdae_fm_output = self.sdae_fm(combined_features)
        dnn_output = self.dnn(combined_features)

        # Applying convolutional networks for Metric Learning
        conv_input = combined_features.unsqueeze(1)  # Adding channel dimension for convolution
        conv_output = F.relu(self.conv1(conv_input))
        conv_output = F.relu(self.conv2(conv_output))
        conv_output = conv_output.view(conv_output.size(0), -1)
        metric_output = self.fc_metric(conv_output)

        # Merging the results of the 3 approaches using a Sigmoid function
        final_output = torch.sigmoid(sdae_fm_output + dnn_output + metric_output)

        return final_output

def train_model(model, data, criterion, optimizer, epochs, user_features=None, occupation_features=None, title_features=None, year_features=None, genre_features=None, item_features=None, scheduler=None, log_file_name=''):
    """
    Trains either Content-Based or Collaborative Filtering model.

    - For Content-Based: Uses user + occupation + title + year + genre features.
    - For Collaborative Filtering: Uses user & item embeddings.
    """     
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user_id, item_id, rating in data:
            optimizer.zero_grad()

            if isinstance(model, CollaborativeFilteringDeepLearning):                
                user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
                item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

                outputs = model(user_feat, item_feat).squeeze()
            else:
                # Extract corresponding feature tensors
                user_feat = torch.tensor(user_features[user_id], dtype=torch.float32)
                occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32)
                title_feat = torch.tensor(title_features[item_id], dtype=torch.float32)
                year_feat = torch.tensor(year_features[item_id], dtype=torch.float32)
                genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32)

                outputs = model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze()

            rating = rating.view(-1)
            outputs = outputs.view(-1)

            loss = criterion(outputs, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        with open("logs/"+log_file_name+"-log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}\n")


        if scheduler:
            scheduler.step(avg_loss)

def evaluate_model(model, test_data, user_features=None, occupation_features=None, title_features=None, year_features=None, genre_features=None, item_features=None, threshold=0.6):
    """
    Evaluates either the Content-Based or Collaborative Filtering model.
    
    - Content-Based: Uses user profile (age, gender, occupation) + item metadata (title, year, genre).
    - Collaborative Filtering: Uses latent embeddings of users and items.
    
    Arguments:
        model: The trained model (either ContentBasedModel or CollaborativeFilteringDeepLearning).
        test_data: Test dataset (list of user-item-rating tuples).
        user_features: Matrix of user profile features (for content-based).
        occupation_features: One-hot encoding of occupations (for content-based).
        item_features: Matrix of item attributes (for both models).
        threshold: Threshold for classification metrics.
    """
      
    model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for user_id, item_id, rating in test_data:
            if isinstance(model, CollaborativeFilteringDeepLearning):
                user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
                item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)
                outputs = model(user_feat, item_feat).squeeze()
            else:
                user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
                occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
                title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
                year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
                genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)                

                outputs = model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze()

            predictions.append(outputs.item())
            ground_truth.append(rating.item())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Prevent NaN values in predictions
    predictions = np.nan_to_num(predictions)

    # Apply threshold for binary classification
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Compute Metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    metrics = {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

    return metrics

def evaluate_hybrid_model_weighted_sum_approach(content_model, collaborative_model, test_data, 
                          user_features, occupation_features, title_features, year_features, genre_features,
                          item_features, alpha=0.6, threshold=0.6):
    """
    Evaluates a hybrid model combining Content-Based and Collaborative Filtering.
    
    Parameters:
        content_model: The trained content-based model.
        collaborative_model: The trained collaborative filtering model.
        test_data: The test dataset (user_id, item_id, rating).
        user_features, occupation_features, title_features, year_features, genre_features: Content-based input features.
        item_features: Features used in collaborative filtering.
        alpha: Weight for content-based model (default: 0.6).
        threshold: Decision threshold for binary classification (default: 0.6).
        
    Returns:
        Dictionary with evaluation metrics.

        Analysis of Content-Based vs Collaborative Filtering
        Your results show:

        Content-Based Model

        Higher Precision (0.6822) and Accuracy (0.6633) → More relevant recommendations.
        Lower Recall (0.7247) → May miss some relevant recommendations.
        Better RMSE (0.2563) and MAE (0.2069) → More precise predictions.
        Collaborative Filtering Model

        High Recall (1.0000) → Captures all relevant recommendations.
        Lower Precision (0.5495) and Accuracy (0.5495) → More false positives.
        Worse RMSE (0.2810) and MAE (0.2363) → Less precise than Content-Based.

        Weighted Sum Approach 

        Assign weights to both models based on performance.
        Example: If Content-Based performs better (higher precision), give it more weight.

        Why This Works Well
        - Balances Both Models: Content-Based is more precise, Collaborative has higher recall → Combining them compensates weaknesses.
        - Weighting Strategy: Adjust (alpha) to control importance:

        alpha = 0.5 → Equal contribution.
        alpha > 0.5 → More weight on Content-Based (better for individual users).
        alpha < 0.5 → More weight on Collaborative Filtering (better for collective recommendations).
        - Adapts to Different Use Cases:
        If the system has many new users, Collaborative Filtering helps more.
        If the system has rich item metadata, Content-Based performs better.
    """
    
    content_model.eval()
    collaborative_model.eval()
    predictions = []
    ground_truth = []
    
    with torch.no_grad():
        for user_id, item_id, rating in test_data:
            # Content-Based Prediction
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            
            # Collaborative Filtering Prediction
            user_feat_cf = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat_cf = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)
            collaborative_pred = collaborative_model(user_feat_cf, item_feat_cf).squeeze().item()

            # Hybrid Prediction: Weighted Sum
            hybrid_pred = alpha * content_pred + (1 - alpha) * collaborative_pred

            predictions.append(hybrid_pred)
            ground_truth.append(rating.item())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Prevent NaN values
    predictions = np.nan_to_num(predictions)

    # Convert predictions to binary using threshold
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Calculate Metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"\n Hybrid Model Evaluation (Threshold: {threshold})")
    print(f" Content-Based Weight: {alpha}, Collaborative Weight: {1 - alpha}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        "threshold": threshold,
        "alpha": alpha,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

def grid_search_hybrid(content_results, collab_results, alphas=np.arange(0.1, 1.0, 0.1), metric="f1score"):
    """
    Automatically finds the best alpha for the hybrid model using grid search.
    
    Parameters:
    - content_results (dict): Metrics from Content-Based Model
    - collab_results (dict): Metrics from Collaborative Filtering Model
    - alphas (list/array): List of weight values to test
    - metric (str): The metric to optimize ("f1score", "precision", "recall", "accuracy", "rmse", "mae")
    
    Returns:
    - best_alpha (float): The best weight for content-based filtering
    - best_metrics (dict): The corresponding metrics for the best hybrid model
    """
    best_alpha = None
    best_score = float('-inf') if metric != "rmse" and metric != "mae" else float('inf')
    best_metrics = None

    print("\n *Running Grid Search for Best Hybrid Weight (alpha)*")
    
    for alpha in alphas:
        beta = 1 - alpha  # Weight for collaborative filtering

        # Compute Hybrid Metrics
        hybrid_metrics = {
            "precision": (alpha * content_results["precision"]) + (beta * collab_results["precision"]),
            "recall": (alpha * content_results["recall"]) + (beta * collab_results["recall"]),
            "f1score": (alpha * content_results["f1score"]) + (beta * collab_results["f1score"]),
            "accuracy": (alpha * content_results["accuracy"]) + (beta * collab_results["accuracy"]),
            "rmse": (alpha * content_results["rmse"]) + (beta * collab_results["rmse"]),
            "mae": (alpha * content_results["mae"]) + (beta * collab_results["mae"]),
        }

        # Check if this alpha is the best one
        if metric in ["rmse", "mae"]:  # Minimize RMSE/MAE
            if hybrid_metrics[metric] < best_score:
                best_score = hybrid_metrics[metric]
                best_alpha = alpha
                best_metrics = hybrid_metrics
        else:  # Maximize Precision, Recall, F1, Accuracy
            if hybrid_metrics[metric] > best_score:
                best_score = hybrid_metrics[metric]
                best_alpha = alpha
                best_metrics = hybrid_metrics

        print(f"Alpha: {alpha:.1f} | {metric.capitalize()}: {hybrid_metrics[metric]:.4f}")

    print(f"\nBest Alpha: {best_alpha:.1f} (Optimized for {metric.capitalize()})")
    print(f"Best Hybrid Metrics: {best_metrics}")

    print("\n**Final Hybrid Model with Optimal Weighting**")
    for key, value in best_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    return best_alpha

def evaluate_hybrid_dynamic_weighting(content_model, collab_model, test_data, 
                                      user_features, occupation_features, title_features, year_features, genre_features, item_features):
    content_model.eval()
    collab_model.eval()

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_id, item_id, rating in test_data:
            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collab_model(user_feat, item_feat).squeeze().item()

            # Compute dynamic weights
            content_error = abs(content_pred - rating.item())
            collab_error = abs(collab_pred - rating.item())
            total_error = content_error + collab_error + 1e-8  # Avoid division by zero

            weight_content = 1 - (content_error / total_error)
            weight_collab = 1 - (collab_error / total_error)

            # Normalize weights
            total_weight = weight_content + weight_collab
            weight_content /= total_weight
            weight_collab /= total_weight

            # Compute final hybrid prediction
            final_prediction = (weight_content * content_pred) + (weight_collab * collab_pred)

            predictions.append(final_prediction)
            ground_truth.append(rating.item())

    # Compute metrics
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    binary_predictions = predictions > 0.6
    binary_ground_truth = ground_truth > 0.6

    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)    
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"**Dynamic Weights Computed:** Content-Based = {weight_content:.4f}, Collaborative = {weight_collab:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        "threshold": 0,
        "precision": precision,
        "recall": recall,
        "f1score": f1,
        "accuracy": accuracy,
        "rmse": rmse,
        "mae": mae
    }

class StackingHybridModel(nn.Module):
    def __init__(self):
        super(StackingHybridModel, self).__init__()
        
        # Meta-model: Simple MLP to combine both predictions
        self.meta_model = nn.Sequential(
            nn.Linear(2, 1),  # Input: [Content-Based Score, Collaborative Score]
            nn.Sigmoid()  # Normalize output as probability
        )

    def forward(self, content_score, collab_score):        
        # Ensure correct dimensions for stacking
        combined = torch.cat((content_score.view(1, -1), collab_score.view(1, -1)), dim=1)

        return self.meta_model(combined)

def evaluate_hybrid_stacking(content_model, collab_model, meta_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features):
    print("\n **Evaluating Hybrid Model (Stacking Meta-Learning)...**")
    
    meta_model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_id, item_id, rating in test_data:

            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collab_model(user_feat, item_feat).squeeze().item()

            # Get content-based prediction
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)

            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze()
            
            # Pass embeddings instead of raw IDs
            collab_pred = collab_model(user_feat, item_feat).squeeze()


            # Pass both predictions through the meta-model
            final_pred = meta_model(content_pred, collab_pred).squeeze()

            predictions.append(final_pred.item())
            ground_truth.append(float(rating))#ground_truth.append(rating.item())


    return compute_metrics(predictions, ground_truth)



def compute_metrics(predictions, ground_truth, threshold=0.6):
    """
    Compute evaluation metrics for the hybrid recommendation model.

    :param predictions: List of predicted scores (continuous values).
    :param ground_truth: List of actual ratings (ground truth).
    :param threshold: The threshold for converting scores to binary recommendations.
    :return: Dictionary containing evaluation metrics.
    """
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Prevent NaN values in predictions
    predictions = np.nan_to_num(predictions)

    # Apply threshold to convert scores into binary classification
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Compute classification metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    
    # Compute error metrics
    mae = mean_absolute_error(ground_truth, predictions)    
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        # Fallback for older versions of sklearn
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    # Print results
    print(f"\n**Hybrid Model Evaluation (Threshold: {threshold})**")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Return as a dictionary for logging
    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

def train_stacking_hybrid_model(meta_model, train_data, content_model, collab_model, 
                                user_features, occupation_features, title_features, 
                                year_features, genre_features, item_features, epochs=10, lr=0.001):
    """
    Trains the Stacking Hybrid Model (Meta-Learning).
    """
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    
    for epoch in range(epochs):
        meta_model.train()
        total_loss = 0

        for user_id, item_id, rating in train_data:
            optimizer.zero_grad()

            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collab_model(user_feat, item_feat).squeeze().item()

            #  Convert user_id and item_id to tensors
            user_id_tensor = torch.tensor([user_id], dtype=torch.long)
            item_id_tensor = torch.tensor([item_id], dtype=torch.long)

            #  Extract user & item features for collaborative filtering model
            user_feat_embed = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat_embed = torch.cat((
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ), dim=1)

            # Fix: Convert NumPy arrays to PyTorch tensors before passing to content model**
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)  
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)    
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)  

            # Predictions from Content-Based Model
            content_pred = content_model(user_feat_embed, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()

            # Predictions from Collaborative Filtering Model
            collab_pred = collab_model(user_feat_embed, item_feat_embed).squeeze().item()

            # Fix: Ensure both predictions are passed as separate arguments**
            meta_output = meta_model(torch.tensor([content_pred], dtype=torch.float32), 
                                     torch.tensor([collab_pred], dtype=torch.float32)).squeeze()

            # Compute loss & update weights
            loss = criterion(meta_output, torch.tensor([rating], dtype=torch.float32))  # Ensure rating is tensor
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1}/{epochs}, Meta Model Loss: {avg_loss:.4f}")


def evaluate_hybrid_bayesian(content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6):
    content_model.eval()
    collaborative_model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_id, item_id, rating in test_data:


            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()


            #  Ensure correct tensor format
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)           

            #  Get predictions from both models
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            
            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Convert predictions to probabilities using Sigmoid
            prob_content = 1 / (1 + np.exp(-content_pred))
            prob_collab = 1 / (1 + np.exp(-collab_pred))

            #  Bayesian Fusion Formula
            final_prediction = (prob_content * prob_collab) / ((prob_content * prob_collab) + ((1 - prob_content) * (1 - prob_collab)))

            predictions.append(final_prediction)
            ground_truth.append(rating.item())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Apply threshold for classification
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Compute evaluation metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)

    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"\n Hybrid Model Evaluation (Bayesian Fusion, Threshold: {threshold})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

def evaluate_hybrid_sigmoid(content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6): 
    content_model.eval()
    collaborative_model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_id, item_id, rating in test_data:

            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Extract feature tensors for content-based model
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
           

            #  Get predictions from both models
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Apply Sigmoid Function
            content_sigmoid = 1 / (1 + np.exp(-content_pred))
            collab_sigmoid = 1 / (1 + np.exp(-collab_pred))

            #  Sigmoid Fusion Formula (Weighted Average)
            final_prediction = (content_sigmoid + collab_sigmoid) / 2  

            predictions.append(final_prediction)
            ground_truth.append(rating.item())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Apply threshold for classification
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Compute evaluation metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)

    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"\nHybrid Model Evaluation (Sigmoid Fusion, Threshold: {threshold})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

def train_random_forest_fusion(content_model, collaborative_model, train_data, user_features, occupation_features, title_features, year_features, genre_features, item_features):
    content_model.eval()
    collaborative_model.eval()
    
    X_train = []
    y_train = []

    with torch.no_grad():
        for user_id, item_id, rating in train_data:

             # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Extract feature tensors for content-based model
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)


            #  Get predictions from both models
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Create training data for RandomForest
            X_train.append([content_pred, collab_pred])
            y_train.append(rating.item())

    #  Convert to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    #  Train the RandomForest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    print("RandomForest model trained successfully!")

    return rf_model


def evaluate_hybrid_random_forest(rf_model, content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6):
    content_model.eval()
    collaborative_model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_id, item_id, rating in test_data:

            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Extract feature tensors for Content-Based Model
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)

            #  Get predictions from both models
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Predict using RandomForest
            final_prediction = rf_model.predict([[content_pred, collab_pred]])[0]

            predictions.append(final_prediction)
            ground_truth.append(rating.item())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Apply threshold for classification
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Compute metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)

    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"\n Hybrid Model Evaluation (RandomForest Fusion, Threshold: {threshold})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

def train_xgboost_fusion(content_model, collaborative_model, train_data, user_features, occupation_features, title_features, year_features, genre_features, item_features):
    content_model.eval()
    collaborative_model.eval()
    
    X = []
    y = []

    with torch.no_grad():
        for user_id, item_id, rating in train_data:

            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Extract feature tensors for Content-Based Model
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)

            #  Get predictions from both models
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Store features and label
            X.append([content_pred, collab_pred])
            y.append(rating.item())

    X = np.array(X)
    y = np.array(y)

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=6)
    xgb_model.fit(X_train, y_train)

    # Evaluate model
    y_pred = xgb_model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"\n **XGBoost Model Trained Successfully! RMSE on Validation Set: {rmse:.4f}**")

    return xgb_model

def evaluate_hybrid_xgboost(xgb_model, content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6):
    content_model.eval()
    collaborative_model.eval()
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for user_id, item_id, rating in test_data:

            # Extract user and item feature vectors
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            item_feat = torch.tensor(item_features[item_id], dtype=torch.float32).unsqueeze(0)

            # Get predictions from both models
            content_pred = content_model(
                user_feat,
                torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0),
                torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)
            ).squeeze().item()

            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Extract feature tensors for Content-Based Model
            user_feat = torch.tensor(user_features[user_id], dtype=torch.float32).unsqueeze(0)
            occ_feat = torch.tensor(occupation_features[user_id], dtype=torch.float32).unsqueeze(0)
            title_feat = torch.tensor(title_features[item_id], dtype=torch.float32).unsqueeze(0)
            year_feat = torch.tensor(year_features[item_id], dtype=torch.float32).unsqueeze(0)
            genre_feat = torch.tensor(genre_features[item_id], dtype=torch.float32).unsqueeze(0)

            #  Get predictions from both models
            content_pred = content_model(user_feat, occ_feat, title_feat, year_feat, genre_feat).squeeze().item()
            collab_pred = collaborative_model(user_feat, item_feat).squeeze().item()

            #  Prepare input for XGBoost
            xgb_input = np.array([[content_pred, collab_pred]])

            #  Predict using trained XGBoost model
            final_prediction = xgb_model.predict(xgb_input)[0]

            predictions.append(final_prediction)
            ground_truth.append(rating.item())

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Apply threshold
    binary_predictions = predictions > threshold
    binary_ground_truth = ground_truth > threshold

    # Compute metrics
    precision = precision_score(binary_ground_truth, binary_predictions, zero_division=0)
    recall = recall_score(binary_ground_truth, binary_predictions, zero_division=0)
    f1 = f1_score(binary_ground_truth, binary_predictions, zero_division=0)
    accuracy = accuracy_score(binary_ground_truth, binary_predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(ground_truth, predictions)
    except ImportError:
        rmse = mean_squared_error(ground_truth, predictions, squared=False)

    print(f"\n **Hybrid Model Evaluation (XGBoost Fusion, Threshold: {threshold})**")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    metrics = {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4)
    }

    return metrics


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path):    

    ratings, movies, users = load_data(database_path)
    ratings, num_users, num_items, user_features, occupation_features, title_features, year_features, genre_features, item_features = preprocess_data(ratings, movies, users)
    
    #Set time of execution start
    start_time = datetime.datetime.now()        

    # Split data for train and test
    # test_size=0.2: Specifies that 20% of the data will go into the test set.
    # random_state=42: A seed for reproducibility. Ensures that the split is the same every time the code run
    train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

    # Prepare data for training 
    train_data = [(torch.tensor([row['user_id']], dtype=torch.long), 
                   torch.tensor([row['item_id']], dtype=torch.long), 
                   torch.tensor([row['rating']], dtype=torch.float32)) for _, row in train_ratings.iterrows()]
    # Prepare data for testing
    test_data = [(torch.tensor([row['user_id']], dtype=torch.long), 
                  torch.tensor([row['item_id']], dtype=torch.long), 
                  torch.tensor([row['rating']], dtype=torch.float32)) for _, row in test_ratings.iterrows()]

    # Define embedding dimensions 
    user_dim = user_features.shape[1]
    occupation_dim = occupation_features.shape[1]
    title_dim = title_features.shape[1] 
    year_dim = year_features.shape[1]
    genre_dim = genre_features.shape[1]
    user_dim = user_features.shape[1] 
    item_dim = item_features.shape[1] 
    embedding_dim = 120        
    
    # Create instance for the models individually
    content_model = ContentBasedModel(user_dim, occupation_dim, title_dim, year_dim, genre_dim, embedding_dim)
    collaborative_model = CollaborativeFilteringDeepLearning(user_dim, item_dim, embedding_dim)

    collaborative_model.apply(init_weights)

    # Define optimizers and loss functions    
    content_optimizer = torch.optim.Adam(content_model.parameters(), lr=0.0001, weight_decay=1e-4)
    collaborative_optimizer = torch.optim.Adam(collaborative_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()    
    
    modo_treino = input('Train content-based deep learning model? [Y|N]: ')    

    if modo_treino == 'N' and os.path.exists(content_model_path):        
        # Load trained model and set evaluation mode
        print("Loading pre-trained model...")
        content_model.load_state_dict(torch.load(content_model_path, weights_only=True))         
        content_model.eval()
    else:
        if os.path.exists(content_model_path):            
            # Load trained model and set trainning mode
            # If already exists trained model, the epochs will be incremented with this new trainning process
            print("Loading pre-trained model...")
            content_model.load_state_dict(torch.load(content_model_path, weights_only=True))            
            content_model.train()  
                    
        epochs = input('Epochs: ')
                
        print("Training Content-Based Model...") 
        log_file_name = algorithm_name + '-ContentBased'        
        train_model( content_model,
            train_data,
            criterion,
            content_optimizer,
            epochs=int(epochs),
            user_features=user_features,
            occupation_features=occupation_features,
            title_features=title_features, 
            year_features=year_features,
            genre_features=genre_features,
            log_file_name=log_file_name
        )        
        # Save trained model
        torch.save(content_model.state_dict(), content_model_path)
        print("Teste-path-content-model: " + content_model_path)
        print("Model saved successfully!")

    modo_treino = input('Train collaborative-filtering deep learning model? [Y|N]: ')    

    if modo_treino == 'N' and os.path.exists(collaborative_model_path):
         # Load trained model and set evaluation mode
        print("Loading pre-trained model...")        
        collaborative_model.load_state_dict(torch.load(collaborative_model_path, weights_only=True))
        collaborative_model.eval()         
    else:

        if os.path.exists(collaborative_model_path):
            # Load trained model and set trainning mode
            # If already exists trained model, the epochs will be incremented with this new trainning process
            print("Loading pre-trained model...")
            collaborative_model.load_state_dict(torch.load(collaborative_model_path, weights_only=True))                        
            collaborative_model.train()  
        
        epochs = input('Epochs: ')

        print("Training Collaborative-Filtering Model...")    
        log_file_name = algorithm_name + '-CollaborativeFiltering'        
        train_model(
            collaborative_model,
            train_data,
            criterion,
            collaborative_optimizer,
            epochs=int(epochs),
            user_features=user_features,
            item_features=item_features,
            log_file_name=log_file_name
        )
        torch.save(collaborative_model.state_dict(), collaborative_model_path)
        print("Model saved successfully!")
            
    # Capture end time of training execution
    end_time = datetime.datetime.now()
    
        
    item_features_dict = {i: item_features[i] for i in range(len(item_features))}

    # Example usage for collaborative filtering
    user_id = 171  # Replace with an actual user ID
    item_ids = list(item_features_dict.keys())  # Extract item IDs

    get_top_n_recommendations(collaborative_model, user_id, user_features, item_features, movies, n=10)

    for test_user in [1, 10, 100, 15, 19, 25]:
        get_top_n_recommendations(collaborative_model, test_user, user_features, item_features, movies, n=5)


    top_100_collab = get_top_N_collaborative(
            model=collaborative_model,
            user_id=user_id,
            item_features=item_features,
            user_features=user_features,
            item_ids=item_ids,
            movies_df=movies,  
            users_df=users,    
            N=100, 
            device='cpu'
        )

    print("Top n Collaborative Filtering Recommendations:")

    top_100_content = get_top_N_content_based(
        model=content_model,
        user_id=user_id,
        item_features=item_features,
        user_features=user_features,
        occupation_features=occupation_features,
        title_features=title_features,
        year_features=year_features,
        genre_features=genre_features,
        item_ids=item_ids,
        movies_df=movies,  
        users_df=users,    
        N=100, 
        device='cpu'
    )   

    #Evaluation
    print("Evaluating Content-Based Model...")     
    log_file_name = algorithm_name + '-ContentBased'
    content_results = evaluate_model(content_model, test_data, user_features, occupation_features, title_features, year_features, genre_features)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, content_results)
     
    log_file_name = algorithm_name + '-CollaborativeFiltering'
    collaborative_results = evaluate_model(collaborative_model, test_data, user_features=user_features, item_features=item_features)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, collaborative_results)

    # Find the best alpha
    best_alpha = grid_search_hybrid(content_results, collaborative_results, alphas=np.arange(0.1, 1.0, 0.1), metric="f1score")            
    log_file_name = algorithm_name + '-Hybrid-weighted_sum_approach'
    print('best_alpha: ')
    print(best_alpha)
    best_alpha = int(best_alpha) #default 0.6  or Adjust weight between Content-Based and Collaborative Filtering
    print("\n Evaluating Hybrid Model...")
    alpha = best_alpha
    results = evaluate_hybrid_model_weighted_sum_approach(
        content_model, collaborative_model, test_data, 
        user_features, occupation_features, title_features, year_features, genre_features,
        item_features, alpha=alpha, threshold=0.6
    )
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, results)
    
    log_file_name = algorithm_name + '-Hybrid_dynamic_weighting'
    print("\n Evaluating Hybrid Model with Dynamic Weighting")
    hybrid_dynamic_weighting = evaluate_hybrid_dynamic_weighting(
    content_model, collaborative_model, test_data, user_features, occupation_features,
    title_features, year_features, genre_features, item_features)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, hybrid_dynamic_weighting)   

    print('\n Bayesian')        
    log_file_name = algorithm_name + '-Hybrid_Bayesian'
    results_hybrid_bayesian = evaluate_hybrid_bayesian(content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, results_hybrid_bayesian)
    
    print('\n Sigmoid Fusion')    
    log_file_name = algorithm_name + '-Hybrid_Sigmoid_Fusion'
    results_hybrid_sigmoid = evaluate_hybrid_sigmoid(content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, results_hybrid_sigmoid)
        
    print('\n RandomForest')    
    log_file_name = algorithm_name + '-Hybrid_RandomForest'
    rf_model = train_random_forest_fusion(content_model, collaborative_model, train_data, user_features, occupation_features, title_features, year_features, genre_features, item_features)
    random_forest_results = evaluate_hybrid_random_forest(rf_model, content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, random_forest_results)
    
    print('\n XgBoost')    
    log_file_name = algorithm_name + '-Hybrid_XgBoost'
    xgb_model = train_xgboost_fusion(content_model, collaborative_model, train_data, user_features, occupation_features, title_features, year_features, genre_features, item_features)
    xgboost_results = evaluate_hybrid_xgboost(xgb_model, content_model, collaborative_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features, threshold=0.6)
    log_execution(log_file_name, start_time, end_time, epochs, embedding_dim, database_test_name, xgboost_results)
    
    print("\n Stacking - Meta-Learning")
    meta_model = StackingHybridModel()       
    log_file_name = algorithm_name + '-Stacking - Meta-Learning'
    train_stacking_hybrid_model(meta_model, train_data, content_model, collaborative_model, user_features, occupation_features, title_features, year_features, genre_features, item_features, epochs = 1)
    stacking_results = evaluate_hybrid_stacking(content_model, collaborative_model, meta_model, test_data, user_features, occupation_features, title_features, year_features, genre_features, item_features)
    log_execution(log_file_name, start_time, datetime.datetime.now(), epochs, embedding_dim, database_test_name, stacking_results)

    return
   
def movieLens10k():
    # MovieLens 10k

    epochs = 1     
    database_test_name = 'movieLens-10k'   
    database_path='datasets/movielens-100k/subset-10k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)    

    epochs = 10       
    database_test_name = 'movieLens-10k'   
    database_path='datasets/movielens-100k/subset-10k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)    

    epochs = 30
    database_test_name = 'movieLens-10k'   
    database_path='datasets/movielens-100k/subset-10k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)    

    epochs = 50
    database_test_name = 'movieLens-10k'   
    database_path='datasets/movielens-100k/subset-10k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)  

def movieLens50k():
    # MovieLens 50k
    epochs = 10       
    database_test_name = 'movieLens-50k'   
    database_path='datasets/movielens-100k/subset-50k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)    
    
    epochs = 30       
    database_test_name = 'movieLens-50k'   
    database_path='datasets/movielens-100k/subset-50k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)    
    
    epochs = 50
    database_test_name = 'movieLens-50k'   
    database_path='datasets/movielens-100k/subset-50k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)  

def movieLens100k():
    # MovieLens 100k
    epochs = 10       
    database_test_name = 'movieLens-100k'   
    database_path='datasets/movielens-100k/converted' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path) 

def movieLens250k():
    # MovieLens 250k
    epochs = 10       
    database_test_name = 'movieLens-250k'   
    database_path='datasets/movielens-1M/subset-250k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)       

def movieLens500k():
    # MovieLens 500k
    epochs = 10       
    database_test_name = 'movieLens-500k'   
    database_path='datasets/movielens-1M/subset-500k' 
    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+".pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"collaborative.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)   

def default_tests_execution():

    print("Select MovieLens dataset for testing:")
    print("[1] MovieLens 10K")
    print("[2] MovieLens 50K")
    print("[3] MovieLens 100K")
    print("[4] MovieLens 250K")
    print("[5] MovieLens 500K")
    print("[6] Exit")
    choice = input("Enter your dataset choice (1-6): ")

    match choice:
        case "1":
            movieLens10k()
        case "2":
            movieLens50k()
        case "3":
            movieLens100k()
        case "4":
            movieLens250k()
        case "5":
            movieLens500k()
        case "6":
            exit()
        case _:
            print("Invalid choice. Please select a number from 1 to 6.")

def custom_test_execution():
        
    epochs = input("Number of Epochs:")

    print("Select MovieLens dataset for testing:")
    print("[1] MovieLens 10K")
    print("[2] MovieLens 50K")
    print("[3] MovieLens 100K")
    print("[4] MovieLens 250K")
    print("[5] MovieLens 500K")
    print("[6] Exit")
    choice = input("Enter your dataset choice (1-6): ")

    match choice:
        case "1":
            database_test_name = 'movieLens-10k'       
            database_path = 'datasets/movielens-100k/subset-10k' 
        case "2":
            database_test_name = 'movieLens-50k'       
            database_path = 'datasets/movielens-100k/subset-50k' 
        case "3":
            database_test_name = 'movieLens-100k'       
            database_path = 'datasets/movielens-100k/converted' 
        case "4":
            database_test_name = 'movieLens-250k'       
            database_path = 'datasets/movielens-1M/subset-250k' 
        case "5":
            database_test_name = 'movieLens-500k'       
            database_path = 'datasets/movielens-1M/subset-500k' 
        case "6":
            exit()
        case _:
            print("Invalid choice. Please select a number from 1 to 5.")

    

    algorithm_name = 'recommendationAlgorithm-v4.1'+'-Epochs'+str(epochs)+'-Base-'+database_test_name
    content_model_path = 'models-data-storage/'+algorithm_name+"_content-based-model.pth"
    collaborative_model_path = 'models-data-storage/'+algorithm_name+"_collaborative-filtering-model.pth"        
    recommendation_principal(algorithm_name, database_test_name, content_model_path, collaborative_model_path, epochs, database_path)        

def main():            

    print("Types of execution: ")
    print("[1] Default tests execution")
    print("[2] Custom tests execution")
    print("[3] Exit")
    choice = input("Enter the type of execution: ")

    match choice:
        case "1":
            default_tests_execution()
        case "2":
            custom_test_execution()
        case "3":
            exit()
        case _:
            print("Invalid choice. Please select a correct type of execution.")    


if __name__ == "__main__":
    main()