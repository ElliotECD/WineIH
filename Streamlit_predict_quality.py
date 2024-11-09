import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from PIL import Image

st.write('''
# White wine quality for professionnals
This application is used to help you determine the quality of your wine
''')

df_w = pd.read_excel("/Final project/winequality-white.xls", engine="xlrd")

# Delete this row if  you do not want to display an image
image = Image.open("/Iron Hack/Vin_blanc.jpeg")
st.image(image, use_column_width=True)

st.write('Amount of wines per quality')
st.line_chart(df_w['quality'].value_counts())

wqual = st.slider("Choose the quality and have a look at the wines' caracteristics", min_value=0, max_value=10,step=1)
st.write('Caracteristics of the wine with a quality equal to : ',wqual, df_w[df_w['quality']==wqual])

st.write('Your wine :')

st.sidebar.header('''
# Caracteristics of your wine
''')

def user_input():
    fixed_a = st.sidebar.slider('fixed acidity :',min_value=1, max_value=15,step=1)
    citric_a = st.sidebar.number_input('citric acide :')
    residual_s = st.sidebar.slider('residual sugar :', min_value=0.5, max_value=70.5, step=0.5)
    chlorides = st.sidebar.number_input('chlorides :')
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide :', min_value=1, max_value=300, step=1)
    total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide :', min_value=1, max_value=450, step=1)
    ph = st.sidebar.slider('ph :', min_value=2.5, max_value=4.5, step=0.1)
    sulphates = st.sidebar.slider('sulphates :', min_value=2.5, max_value=4.5, step=0.1)
    alcohol = st.sidebar.slider('alcohol :', min_value=5, max_value=15, step=1)
    data = {'fixed acidity': fixed_a,
            'citric acid': citric_a,
            'residual sugar': residual_s,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'pH': ph,
            'sulphates': sulphates,
            'alcohol': alcohol}
    wine_param = pd.DataFrame(data, index=[0])
    return wine_param

df = user_input()

st.write(df)

X = df_w[['fixed acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'pH', 'sulphates', 'alcohol']]
y = df_w['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_num = X_train.select_dtypes(np.number)
X_test_num = X_test.select_dtypes(np.number)

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

prediction = clf.predict(df.select_dtypes(np.number))

st.write('The prediction :')
st.write(prediction)
