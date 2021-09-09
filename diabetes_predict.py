import numpy as np
import streamlit as st
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@st.cache()
def d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):
    # Split the train and test dataset.
    feature_columns = list(diabetes_df.columns)

    # Remove the 'Pregnancies', Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Outcome')

    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree_clf.fit(X_train, y_train)
    y_train_pred = dtree_clf.predict(X_train)
    y_test_pred = dtree_clf.predict(X_test)
    # Predict diabetes using the 'predict()' function.
    prediction = dtree_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)

    return prediction, score


def grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):
    feature_columns = list(diabetes_df.columns)
    # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Outcome')
    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    # Split the train and test dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(4, 21), 'random_state': [42]}

    # Create a grid
    grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='roc_auc', n_jobs=-1)

    # Training
    grid_tree.fit(X_train, y_train)
    best_tree = grid_tree.best_estimator_

    # Predict diabetes using the 'predict()' function.
    prediction = best_tree.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(grid_tree.best_score_ * 100, 3)

    return prediction, score


def app(diabetes_df):
    st.markdown(
        "<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier</b> for the Early Prediction of Diabetes.",
        unsafe_allow_html=True)
    st.subheader("Select Values:")

    glucose = st.slider('Select Glucose values', int(diabetes_df['Glucose'].min()), int(diabetes_df['Glucose'].max()),
                        1)
    bp = st.slider('Select BP Value', int(diabetes_df['Blood_Pressure'].min()),
                   int(diabetes_df['Blood_Pressure'].max()), 1)
    insulin = st.slider('Select insulin Value', int(diabetes_df['Insulin'].min()), int(diabetes_df['Insulin'].max()), 1)
    bmi = st.slider('Select BMI Value', int(diabetes_df['BMI'].min()), int(diabetes_df['BMI'].max()), 1)
    pedigree = st.slider('Select Pedigree Function Value', int(diabetes_df['Pedigree_Function'].min()),
                         int(diabetes_df['Pedigree_Function'].max()), 1)
    age = st.slider('Select Age', int(diabetes_df['Age'].min()), int(diabetes_df['Age'].max()), 1)

    st.subheader("Model Selection")

    # Add a single select drop down menu with label 'Select the Classifier'
    predictor = st.selectbox("Select the Decision Tree Classifier",
                             ('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier'))

    if predictor == 'Decision Tree Classifier':
        if st.button("Predict"):
            prediction, score = d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The accuracy score of this model is", score, "%")
    elif predictor == 'GridSearchCV Best Tree Classifier':
        if st.button("Predict"):
            prediction, score = grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Optimised Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The best score of this model is", score, "%")
