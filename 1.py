import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd  
from sklearn.feature_extraction.text import TfidfVectorizer  
# Import BeautifulSoup for web scraping
from bs4 import BeautifulSoup  
import random
df = pd.read_csv("City.csv")

# Combine relevant features into a single text column for TF-IDF vectorization
df['combined_features'] = df['Best Time'] + ' ' + df['City_desc']

# Use TF-IDF vectorizer to convert the text data into a matrix of TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])
# Function to search and display images based on a query using Google Images
def search_and_display_images(query, num_images=20):
    try:
        # Initialize an empty list for image URLs
        k=[]  
        # Initialize an index for iterating through the list of images
        idx=0  
        # Construct Google Images search URL
        url = f"https://www.google.com/search?q={query}&tbm=isch"  
         # Make an HTTP request to the URL
        response = requests.get(url) 
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")  
        # Initialize an empty list for storing image URLs
        images = []  
        # Iterate through image tags in the HTML content
        for img in soup.find_all("img"):  
             # Limit the number of images to the specified amount
            if len(images) == num_images: 
                break
            # Get the image source URL
            src = img.get("src")  
            # Check if the source URL is valid
            if src.startswith("http") and not src.endswith("gif"):  
                # Add the image URL to the list
                images.append(src)  
        # Iterate through the list of image URLs
        for image in images:  
            # Add each image URL to the list 'k'
            k.append(image)  
        # Reset the index for iterating through the list of image URLs
        idx = 0  
        # Iterate through the list of image URLs
        while idx < len(k):
            # Iterate through the columns in a 4-column layout 
            for _ in range(len(k)): 
                # Create 4 columns for displaying images 
                cols = st.columns(4)  
                # Display the first image in the first column
                cols[0].image(k[idx], width=150)  
                idx += 1 
                # Move to the next image in the list
                cols[1].image(k[idx], width=150)
                # Display the second image in the second column
                idx += 1  
                # Move to the next image in the list
                cols[2].image(k[idx], width=150)  
                # Display the third image in the third column
                idx += 1  
                # Move to the next image in the list
                cols[3].image(k[idx], width= 150)  
                # Display the fourth image in the fourth column
                idx = idx + 1  
                # Move to the next image in the list
    except:
         # Handle exceptions gracefully if there is an error while displaying images
        pass  
   

# Function to get recommendations based on user input
def get_recommendations(user_best_time):
    # Combine user input into a similar text format
    user_input = user_best_time
    
    # Transform user input using the TF-IDF vectorizer
    user_tfidf = tfidf_vectorizer.transform([user_input])
    
    # Calculate dot product between user input and existing items
    similarity_scores = user_tfidf.dot(tfidf_matrix.T)
    
    # Get indices of items with positive similarity scores
    similar_indices = similarity_scores.indices
    
    # Get recommendations with description
    recommendations = df.loc[similar_indices, ['City', 'City_desc']]
    # returning the recommendations
    return recommendations

st.set_page_config(page_title="Destination Recommender", page_icon=":hospital:")


user_best_time = st.selectbox("Select the best time to visit", df['Best Time'].unique())
recommendations = get_recommendations(user_best_time)

# Display recommendations in a dropdown
selected_recommendation = st.selectbox("Select a destination", recommendations['City'].tolist())
genre = st.radio(
"Select Option:",
[":rainbow[Place]", ":rainbow[Near Places]", ":rainbow[Accommodations]"])

if genre == ":rainbow[Place]":   

    # Display selected recommendation and its description
    st.subheader("Selected Recommendation:")
    # Display the selected recommendation
    selected_description = recommendations.loc[recommendations['City'] == selected_recommendation, 'City_desc'].iloc[0]
    # Display the description of the selected recommendation
    st.write(f"**{selected_recommendation}**: {selected_description}")
    # Display the selected recommendation
    search_and_display_images(selected_recommendation)
if genre == ":rainbow[Near Places]":
    # Read Places dataset from CSV
    df1=pd.read_csv("Places.csv")
    # Display the nearest places
    st.subheader("Nearest Places:")
    # Display the nearest places
    res=df1.loc[df1['City'] == selected_recommendation]
    # Display the nearest places
    for index, row in res.iterrows():
        st.write( row['Place'])
        st.write("Distance: ",row['Distance'])
        st.write(row['Place_desc'])
        search_and_display_images(row['Place'],4)
if genre == ":rainbow[Accommodations]":
    # Read Travel Cost dataset from CSV
    df2=pd.read_csv('travel cost.csv')
    # Display the accommodation cost
    st.subheader("Accommodation Cost:")
    # Display the accommodation cost
    res1=df2.loc[df2['City'] == selected_recommendation]
    # get the accommodation cost for the selected recommendation
    for index, row in res1.iterrows():
        st.write( row['Accomadation_Type'],":",row['Accomdation_Cost'])
        text= row['Accomadation_Type']+"in"+selected_recommendation
        st.write(row['source'])
        search_and_display_images(text,4)

