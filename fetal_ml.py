# App to predict fetal health class
# Using a pre-trained ML RandomForestClassifier model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle

st.title('Fetal Health Classification: A Machine Learning App') 

# Display the image
st.image('fetal_health_image.gif', width = 400)

st.subheader('Utilize our advanced Machine Learning application to predict fetal health classifications.') 

# Reading the pickle files that we created before 
rf_pickle = open('random_forest_fetal.pickle', 'rb') 
map_pickle = open('output_fetal.pickle', 'rb') 
clf = pickle.load(rf_pickle) 
unique_fetal_mapping = pickle.load(map_pickle) 
rf_pickle.close() 
map_pickle.close() 

# Define a function to color cells based on class
def color_cells(val):
    if val == "Normal":
        return 'background-color: rgba(50, 205, 50, 0.75)' #limegreen
    elif val == "Suspect":
        return 'background-color: rgba(255, 255, 0, 0.75)' #yellow
    elif val == "Pathological":
        return 'background-color: rgba(255, 165, 0, 0.75)' #orange
    else:
        return ''

# Loading original data
original_df = pd.read_csv('fetal_health.csv') # Original data to create ML model

# Remove output from original data
original_df = original_df.drop(columns = ['fetal_health'])

# Show sample dataframe
st.write("To ensure optimal results, please ensure that your data strictly adheres to the specified format outlined below:") 
st.dataframe(original_df.head())

# Ask users to input their data as a file
fetal_file = st.file_uploader('Upload your own fetal health CTG file')



if fetal_file is not None:
    # Loading user data
    user_df = pd.read_csv(fetal_file) # User provided data

    # Dropping null values
    user_df = user_df.dropna() 
    original_df = original_df.dropna() 

    # Ensure the order of columns in user data is in the same order as that of original data
    user_df = user_df[original_df.columns]

    # Concatenate two dataframes together along rows (axis = 0)
    combined_df = pd.concat([original_df, user_df], axis = 0)

    # Number of rows in original dataframe
    original_rows = original_df.shape[0]

    # Create dummies for the combined dataframe
    combined_df_encoded = pd.get_dummies(combined_df)

    # Split data into original and user dataframes using row index
    original_df_encoded = combined_df_encoded[:original_rows]
    user_df_encoded = combined_df_encoded[original_rows:]

    # Predictions for user data
    user_pred = clf.predict(user_df_encoded)

    # Predicted species
    user_pred_class = unique_fetal_mapping[user_pred]

    # Adding predicted species to user dataframe
    user_df['Predicted Fetal Health'] = user_pred_class



    # Map numerical class values to class names
    class_mapping = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}
    user_df['Predicted Fetal Health'] = user_df['Predicted Fetal Health'].map(class_mapping)

    # Prediction Probabilities
    user_pred_prob = clf.predict_proba(user_df_encoded)
    # Storing the maximum prob. (prob. of predicted species) in a new column
    user_df['Predicted Probability %'] = user_pred_prob.max(axis = 1)

    # Apply the styling to the DataFrame
    result = user_df.style.applymap(color_cells, subset=['Predicted Fetal Health'])


    # Show the predicted species on the app
    st.subheader("Predicting Fetal Health")
    st.dataframe(result)
else:
    st.write("Upload a CSV file!")

# Showing additional items
st.subheader("Prediction Performance")
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

with tab1:
  st.image('feature_imp.svg')
with tab2:
  st.image('confusion_mat.svg')
with tab3:
    df = pd.read_csv('class_report.csv', index_col=0)
    st.dataframe(df)