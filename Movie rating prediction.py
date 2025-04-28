#!/usr/bin/env python
# coding: utf-8

# # Movie Rating Prediction

# # Tasks to perform:
#    ### • Build a predictive model to estimate movie ratings based on different attributes. 
#    ### • Perform data preprocessing, including encoding categorical variables and handling missing values. 
#    ### • Engineer useful features like director success rate and average rating of similar movies. 
#    ### • Evaluate the model using appropriate techniques. 
#    ### • Expected outcome: A model that accurately predicts movie ratings based on given inputs. 
#    ### • Submit a structured GitHub repository with documentation on approach, preprocessing, and performance evaluation.

# # **Steps to perform above task:**
# #### 1. Load Data
# #### 2. Exploratory Data Analysis (EDA) and Initial Cleaning
# #### 3. Preprocessing (Handling Missing Values, Cleaning Columns, Encoding)
# #### 4. Feature Engineering (Director/Genre Averages)
# #### 5. Train-Test Split
# #### 6. Model Training
# #### 7. Model Evaluation
# #### 8. Conclusion

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# # 1. Loading data

# In[2]:


data = pd.read_csv(r"C:\Users\dell\Downloads\movie project data.csv", encoding = 'cp1252')


# In[3]:


print("Shape of the dataset we loaded:", data.shape)
print("First 5 Rows of our dataset:",)
print(data.head(10))
print("Dataset Information:")
data.info()


# # 2. Initial Cleaning

# In[4]:


print("Missing values we have before cleaning:")
print(data.isnull().sum())


# In[5]:


data.dropna(subset=['Rating'], inplace=True)
print(f"Shape of dataset after dropping rows with missing Rating: {data.shape}")
print("Missing values we left with after dropping missing Ratings:")
print(data.isnull().sum())


# In[6]:


plt.figure(figsize=(10, 5))
sns.histplot(data['Rating'], kde=True, bins=20)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

print("\nRating Description:")
print(data['Rating'].describe())


# ## Cleaning Specific Columns
# year, duration, and votes

# In[7]:


data['Year'] = data['Year'].astype(str).str.extract(r'(\d{4})', expand=False)
data['Year'] = pd.to_numeric(data['Year'], errors='coerce') 


# In[8]:


data['Duration'] = data['Duration'].astype(str).str.replace(' min', '', regex=False)
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')


# In[9]:


data['Votes'] = data['Votes'].astype(str).str.replace(',', '', regex=False)
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')


# In[10]:


print("After cleaning Year, Duration, Votes we have the following data types:")
data.info()


# In[11]:


numerical_impute = ['Year', 'Duration', 'Votes']
categorical_impute = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']


# ## Imputing numerical columns with median

# In[12]:


num_imputer = SimpleImputer(strategy='median')
data[numerical_impute] = num_imputer.fit_transform(data[numerical_impute])


# ## Imputing categorical columns with 'Unknown'

# In[13]:


for col in categorical_impute:
    data[col].fillna('Unknown', inplace=True)


# In[14]:


print("Missing Values we have after imputation of both the categorical and numerical columns:")
print(data.isnull().sum())


# # Top N Directors for Interactive Options

# In[15]:


print("Identifying Top Directors")
N_TOP_DIRECTORS = 100 
top_n_directors_list = [] 


if 'Director' in data.columns:
    
    director_counts = data[data['Director'] != 'Unknown']['Director'].value_counts()
    if not director_counts.empty:
        top_n_directors_list = director_counts.head(N_TOP_DIRECTORS).index.tolist()
        print(f"Identified Top {len(top_n_directors_list)} Directors for interactive options.")
        
    else:
        print("No valid (non-'Unknown') director names found to create top list.")
else:
    print("Warning: 'Director' column not found in DataFrame 'df'. Cannot identify top directors.")


# # 3. Preprocessing (Handling Missing Values, Cleaning Columns, Encoding)

# In[16]:


data['Primary_Genre'] = data['Genre'].apply(lambda x: x.split(', ')[0] if isinstance(x, str) else 'Unknown')
print("Top 10 Primary Genres we have in our data:")
print(data['Primary_Genre'].value_counts().head(10))


# In[17]:


numerical_cols = ['Year', 'Duration', 'Votes']
categorical_cols = ['Primary_Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'] 
cate_cols_to_drop = ['Director', 'Actor 1', 'Actor 2', 'Actor 3']
cat_col_ohe = 'Primary_Genre'
target_col = 'Rating'


# # Drop original 'Genre' and 'Name'
# #### here we are using a new dataframe to avoid modification in our original dataframe

# In[18]:


data_processed = data.drop(['Genre', 'Name'], axis=1) 


# # 3. Training and testing data splitting  

# In[19]:


X = data_processed.drop(target_col, axis=1)
y = data_processed[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of Training set: X={X_train.shape}, y={y_train.shape}")
print(f"Shape of Testing set: X={X_test.shape}, y={y_test.shape}")

X_train


# # 4. Preprocessing & Feature Engineering

# In[20]:


train_data_temp = X_train.copy()
train_data_temp['Rating'] = y_train
train_data_temp


# # Director Average Rating

# In[21]:


director_avg_rating_map = train_data_temp.groupby('Director')['Rating'].mean()
X_train['Director_Avg_Rating'] = X_train['Director'].map(director_avg_rating_map)
X_test['Director_Avg_Rating'] = X_test['Director'].map(director_avg_rating_map)


# # Primary Genre Average Rating

# In[22]:


genre_avg_rating_map = train_data_temp.groupby(cat_col_ohe)['Rating'].mean()
X_train['Genre_Avg_Rating'] = X_train[cat_col_ohe].map(genre_avg_rating_map)
X_test['Genre_Avg_Rating'] = X_test[cat_col_ohe].map(genre_avg_rating_map)


# In[23]:


train_mean_rating = y_train.mean()

X_train['Director_Avg_Rating'].fillna(train_mean_rating, inplace=True)
X_train['Genre_Avg_Rating'].fillna(train_mean_rating, inplace=True)
X_test['Director_Avg_Rating'].fillna(train_mean_rating, inplace=True)
X_test['Genre_Avg_Rating'].fillna(train_mean_rating, inplace=True)


# # Encoding (for only training data)
#  
# for converting remaining categorical features to numerical format.
# 

# In[24]:


ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(X_train[[cat_col_ohe]])


# # Transforming and creating DataFrames

# In[25]:


genre_encoded_train = pd.DataFrame(ohe.transform(X_train[[cat_col_ohe]]), 
                                   columns=ohe.get_feature_names_out([cat_col_ohe]), index=X_train.index)
genre_encoded_test = pd.DataFrame(ohe.transform(X_test[[cat_col_ohe]]), 
                                  columns=ohe.get_feature_names_out([cat_col_ohe]), index=X_test.index)


# # Joining encoded columns back

# In[26]:


X_train = X_train.join(genre_encoded_train)
X_test = X_test.join(genre_encoded_test)


# # Droping original categorical columns used for encoding/feature engineering

# In[27]:


cols_to_drop = [cat_col_ohe] + cate_cols_to_drop
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)


# # Scaling Numerical Features 

# In[28]:


cols_to_scale = numerical_cols + ['Director_Avg_Rating', 'Genre_Avg_Rating']


# In[29]:


#scaling only those columns which are present in both sets 
common_cols_to_scale = list(set(cols_to_scale) & set(X_train.columns) & set(X_test.columns))


# In[30]:


scaler = StandardScaler()
X_train[common_cols_to_scale] = scaler.fit_transform(X_train[common_cols_to_scale])
X_test[common_cols_to_scale] = scaler.transform(X_test[common_cols_to_scale])  

missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0
X_test = X_test[X_train.columns]


# In[31]:


print("Shape of training data after preprocessing/engineering:", X_train.shape)
print("Testing data shape after preprocessing/engineering:", X_test.shape)
print(f"Number of features: {X_train.shape[1]}")


# # Ensuring Column Consistency 

# In[33]:


train_final_cols = X_train.columns.tolist()
X_test = X_test.reindex(columns=train_final_cols, fill_value=0)


print("Preprocessing is completed.")
print(f"Final number of features for training are: {X_train.shape[1]}")


# In[34]:


print("Initializing Random Forest Model  ")

rf_model = RandomForestRegressor(n_estimators=100, 
                                 random_state=42,  
                                 n_jobs=-1,        
                                 max_depth=15,     
                                 min_samples_leaf=5,
                                 oob_score=True) 


# # Training the model on training dataset:

# In[35]:


rf_model.fit(X_train, y_train)

print("Model training is completed.")


# # Evaluating the model on testing dataset:

# In[36]:


print("Evaluating Random Forest Model on Test Set  ")
y_pred_test = rf_model.predict(X_test)


# # Calculating the standard regression metrics

# In[37]:


mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)


# In[38]:


print(f"Test Set Performance:")
print(f"  Mean Absolute Error (MAE):  {mae_test:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_test:.4f}")
print(f"  R-squared (R²):             {r2_test:.4f}")


# # 8. Interactive Prediction Section 
# ## Prepare necessary lists and helper functions for the interactive prediction loop.

# In[39]:


print("Preparing for Interactive Prediction")
try:
    known_genres = list(ohe.categories_[0])
    print(f"Detected {len(known_genres)} genres from training data for selection.")
except NameError:
    print("ERROR: OneHotEncoder object 'ohe' not found. Cannot proceed with genre options.")
    known_genres = ['Drama', 'Action', 'Comedy', 'Unknown']
except Exception as e:
    print(f"Warning: Could not extract genres from OHE. Using fallback. Error: {e}")
    known_genres = ['Drama', 'Action', 'Comedy', 'Thriller', 'Romance', 'Unknown'] 


try:
    imputer_medians = num_imputer.statistics_
    median_map = {col: med for col, med in zip(numerical_impute, imputer_medians)}
    print("Detected medians for numerical input defaults.")
except NameError:
    print("ERROR: SimpleImputer object 'num_imputer' or column list 'numerical_impute' not found. Using fallback defaults.")
    median_map = {'Year': 2015.0, 'Duration': 130.0, 'Votes': 1000.0} # Fallback
except Exception as e:
    print(f"Warning: Could not extract medians from imputer. Using fallback. Error: {e}")
    median_map = {'Year': 2015.0
                  , 'Duration': 130.0, 'Votes': 1000.0}



# # Calculating Top N Directors list 
# 
# and here we are using the cleaned DataFrame "data"

# In[40]:


print("Identifying Top Directors for options")
N_TOP_DIRECTORS = 100 # showing 100 option for director feature
top_n_directors = [] 
try:
    if 'data' in globals() and 'Director' in data.columns:
        director_counts = data[data['Director'] != 'Unknown']['Director'].value_counts()
        if not director_counts.empty:
            top_n_directors = director_counts.head(N_TOP_DIRECTORS).index.tolist()
            print(f"Identified Top {len(top_n_directors)} Directors for interactive options.")
        else:
            print("No valid director names are found to create top list.")
    else:
        print("Warning: Cleaned DataFrame 'data' or 'Director' column not found. Cannot identify top directors.")
except Exception as e:
    print(f"An error occurred identifying top directors: {e}")


# # Helper Functions for Interactive UserInput
# 

# In[41]:


def numerical_input(prompt, col_name):
    median_val = median_map.get(col_name, 0)
    while True:
        try:
            value_str = input(f"{prompt} (Press Enter for default={median_val:.0f}): ").strip()
            if not value_str:
                 print(f"  -> Using default value: {median_val:.0f}")
                 return float(median_val)
            value = float(value_str)
            return value
        except ValueError:
            print("  Invalid input. Please enter a number.")



# In[42]:


def categorical_input(prompt, options):
    print(f"{prompt}")
    display_options = sorted([opt for opt in options if opt != 'Unknown']) + (['Unknown'] if 'Unknown' in options else [])
    if not display_options:
         print("Warning: No options provided for selection!!!")
         return "Unknown"
    for i, option in enumerate(display_options):
        print(f"  {i + 1}. {option}")
    while True:
        try:
            choice_str = input("Enter the number of your choice: ").strip()
            if not choice_str:
                print("  Input required. Please select a number.")
                continue
            choice = int(choice_str)
            if 1 <= choice <= len(display_options):
                selected_option = display_options[choice - 1]
                print(f"  -> Selected: {selected_option}")
                return selected_option
            else:
                print(f"  Invalid choice. Please enter a number between 1 and {len(display_options)}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


# In[43]:


def director_input(prompt, top_directors_list):
    print(f"{prompt}")
    if not top_n_directors_list: 
        print(" (Top director list not available, please enter name manually)")
        while True:
            manual_director = input("  Please enter the Director's Name: ").strip()
            if manual_director:
                return manual_director
            else:
                print("    Director's name cannot be empty.")
    
    options_with_other = sorted(top_directors_list) + ["(Other - Enter Manually)"]
    for i, option in enumerate(options_with_other):
        print(f"  {i + 1}. {option}")
    while True:
        try:
            choice_str = input(f"Select a director (1-{len(options_with_other)}) or choose 'Other': ").strip()
            if not choice_str:
                print("  Input required. Please select a number from the list above.")
                continue
            choice = int(choice_str)
            if 1 <= choice < len(options_with_other):
                selected_director = options_with_other[choice - 1]
                print(f"  -> Selected: {selected_director}")
                return selected_director
            elif choice == len(options_with_other):
                print("  Selected 'Other'.")
                while True:
                    manual_director = input("  Please enter the Director's Name: ").strip()
                    if manual_director:
                        return manual_director
                    else:
                        print("    Director's name cannot be empty if selecting 'Other'.")
            else:
                print(f"  Invalid choice. Please enter a number between 1 and {len(options_with_other)}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


# # Now we are making prediction function using In-Memory objects

# In[44]:


def predict_rating(movie_features):
    
    print(f"  Preprocessing Input and Predicting  ")
    try:
        required_objects = ['num_imputer', 'numerical_impute', 'director_avg_rating_map',
                           'genre_avg_rating_map', 'train_mean_rating', 'ohe', 'cat_col_ohe',
                           'cate_cols_to_drop', 'scaler', 'common_cols_to_scale',
                           'train_final_cols', 'rf_model']
        for obj_name in required_objects:
            if obj_name not in globals():
                raise NameError(f"Essential object '{obj_name}' is not defined. Prediction cannot proceed.")

        
        new_data = pd.DataFrame([movie_features])

        
        for i, col in enumerate(numerical_impute): # Use YOUR list name
            if col in new_data.columns:
                new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
                if new_data[col].isnull().any():
                    fill_value = num_imputer.statistics_[i]
                    print(f"  Note: Filling missing/invalid '{col}' with median: {fill_value:.0f}")
                    new_data[col].fillna(fill_value, inplace=True)
            else:
                 fill_value = num_imputer.statistics_[i]
                 print(f"  Warning: Required numeric column '{col}' missing. Using median: {fill_value:.0f}")
                 new_data[col] = fill_value

       
        new_data['Director_Avg_Rating'] = new_data['Director'].map(director_avg_rating_map).fillna(train_mean_rating)
        new_data['Genre_Avg_Rating'] = new_data['Primary_Genre'].map(genre_avg_rating_map).fillna(train_mean_rating)

        
        genre_encoded_new = pd.DataFrame(ohe.transform(new_data[[cat_col_ohe]]),
                                         columns=ohe.get_feature_names_out([cat_col_ohe]),
                                         index=new_data.index)
        new_data = new_data.join(genre_encoded_new)

        
        original_cols_to_drop = [cat_col_ohe] + cate_cols_to_drop
        cols_to_drop_existing = [col for col in original_cols_to_drop if col in new_data.columns]
        new_data = new_data.drop(columns=cols_to_drop_existing)

        
        cols_to_scale_existing = [col for col in common_cols_to_scale if col in new_data.columns]
        if cols_to_scale_existing:
            new_data[cols_to_scale_existing] = scaler.transform(new_data[cols_to_scale_existing])


        final_input_df = new_data.reindex(columns=train_final_cols, fill_value=0)

       
        predicted_rating = rf_model.predict(final_input_df)
        return predicted_rating[0]

    except NameError as ne:
         print(f"ERROR: A required object was not found in memory: {ne}")
         print("Please ensure all preprocessing cells have been run successfully BEFORE this section.")
         return None
    except Exception as e:
        print(f"ERROR during prediction pipeline: {e}")
        traceback.print_exc()
        return None


# # Get User Input for Predict 

# In[45]:


print("\n" + "="*50)
print("      INTERACTIVE MOVIE RATING PREDICTOR      ")
print("="*50)
print("Enter the details for the movie you want to predict.")
print("(Press Enter at numerical prompts to use default values based on training data medians)")


essential_vars_exist = True

required_vars_check = ['num_imputer', 'numerical_impute', 'director_avg_rating_map',
                 'genre_avg_rating_map', 'train_mean_rating', 'ohe', 'cat_col_ohe',
                 'cate_cols_to_drop', 'scaler', 'common_cols_to_scale',
                 'train_final_cols', 'rf_model', 'top_n_directors', 'known_genres',
                 'median_map']
for var_name in required_vars_check:
    if var_name not in globals():
        print(f"ERROR: Essential variable '{var_name}' for prediction not defined. Cannot start loop.")
        essential_vars_exist = False

if essential_vars_exist:
    while True:
       
        input_features = {}
        print("\n" + "-"*20 + " Enter Movie Details " + "-"*20)
        
        input_features['Year'] = numerical_input("Enter Year", numerical_impute[0])
        input_features['Duration'] = numerical_input("Enter Duration (in minutes)", numerical_impute[1])
        input_features['Votes'] = numerical_input("Enter approximate Number of Votes", numerical_impute[2])
       
        input_features['Director'] = director_input("Select or Enter the Director:", top_n_directors)
        
        input_features['Primary_Genre'] =categorical_input("Select the Primary Genre:", known_genres)

      
        prediction = predict_rating(input_features)

        
        print("-" * 20 + " Prediction Result " + "-"*20)
        if prediction is not None:
            print(f"🎬 Predicted IMDb Rating: {prediction:.2f}")
        else:
            print("⚠️ Could not generate a prediction due to an error during processing.")
        print("-" * 55)

        
        another = input("\nPredict another movie? (yes/no): ").strip().lower()
        if another != 'yes':
            break

    print("Exiting interactive predictor.")
else:
    print("Interactive prediction cannot start due to missing essential variables.")


# In[ ]:




