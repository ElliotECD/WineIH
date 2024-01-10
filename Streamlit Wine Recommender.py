import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from PIL import Image

# Load wine data
df_w = pd.read_excel("/Users/accountmanager/Desktop/Iron Hack/Final project/winequality-white.xls", engine="xlrd")
df_w.insert(0, 'id', range(1, len(df_w) + 1))

st.title("White Wine Recommender")

# Display user input options
st.sidebar.title("Select your preferences")
fixed_acidity = st.sidebar.slider("Fixed Acidity", float(df_w['fixed acidity'].min()), float(df_w['fixed acidity'].max()), (float(df_w['fixed acidity'].min()), float(df_w['fixed acidity'].max())))
residual_sugar = st.sidebar.slider("Residual Sugar", float(df_w['residual sugar'].min()), float(df_w['residual sugar'].max()), (float(df_w['residual sugar'].min()), float(df_w['residual sugar'].max())))
alcohol = st.sidebar.slider("Alcohol", float(df_w['alcohol'].min()), float(df_w['alcohol'].max()), (float(df_w['alcohol'].min()), float(df_w['alcohol'].max())))

# Filter wines based on user input
filtered_wines = df_w[(df_w['fixed acidity'] >= fixed_acidity[0]) & (df_w['fixed acidity'] <= fixed_acidity[1]) & (df_w['residual sugar'] >= residual_sugar[0]) & (df_w['residual sugar'] <= residual_sugar[1]) & (df_w['alcohol'] >= alcohol[0]) & (df_w['alcohol'] <= alcohol[1])]

if filtered_wines.empty:
    st.warning("No wines found with your selected criteria. Please adjust your preferences.")
else:
    st.subheader("Wines matching your preferences")
    st.dataframe(filtered_wines)

    image = Image.open("/Users/accountmanager/Desktop/vinoblanco.jpeg")
    st.image(image, use_column_width=True)

"Just Below, another solution to help you find something new based on something you already know this time, you can enter the ID of a wine that you already know and it will give you an ID of another wine"

    # Allow user to select a wine ID
wine_id = st.number_input("Select a wine ID to get a recommendation", value=int(filtered_wines['id'].min()),
                              min_value=int(filtered_wines['id'].min()), max_value=int(filtered_wines['id'].max()))


def get_recommendation(wine_id):
    X = df_w.drop(['id', 'quality'], axis=1)
    # y = df['artist']

    scaler = StandardScaler()

    X_prep = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_prep)

    clusters = kmeans.predict(X_prep)

    scaled_df = pd.DataFrame(X_prep, columns=X.columns)
    scaled_df['id'] = df_w['id']
    scaled_df['cluster'] = clusters
    scaled_df = scaled_df.round(3)

    results = df_w.loc[df_w['id'] == wine_id]

    df_ = pd.DataFrame(results)
    new_features = df_[X.columns]

    scaled_x = scaler.transform(new_features)
    cluster = kmeans.predict(scaled_x)

    filtered_df = np.array(scaled_df[scaled_df['cluster'] == cluster[0]][X.columns], order="C")
    closest, _ = pairwise_distances_argmin_min(scaled_x, filtered_df)

    st.write('Based on your selection, we recommend trying this wine which is similar to the one you chose:')
    st.write(scaled_df.loc[closest]['id'])


if st.button('Get recommendation'):
    result = get_recommendation(wine_id)




