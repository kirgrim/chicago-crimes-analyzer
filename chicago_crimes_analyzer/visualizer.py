import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from processor import data_processor


def main():
    st.title("Chicago Crimes in 2024")
    data_processor.reset_filters()

    st.markdown('##')

    st.title('Thesis #1')
    null_hypothesis = f'There is no significant association between the crime location (latitude and longitude) and the likelihood of an arrest.'
    alternative_hypothesis = f'There is a significant association between the crime location (latitude and longitude) and the likelihood of an arrest.'
    run_attribute_analysis(null_h=null_hypothesis, alter_h=alternative_hypothesis, column_x=['Latitude', 'Longitude'],
                           column_y='Arrest', alpha_score=0.05, convert_data=False)
    df_copy = data_processor.data.copy()
    df_copy['Arrest'] = df_copy['Arrest'].astype(int)
    model = smf.logit(formula='Arrest ~ Latitude + Longitude', data=df_copy).fit()
    st.write(model.summary())

    st.write(r"$\textsf{\Large Map of Chicago Crimes:}$")
    precision = st.slider("Adjust coordinates precision with slider", 1, 5, 2)
    st.map(data_processor.df_grouped_by_coordinates(precision=precision),
           latitude='Latitude',
           longitude='Longitude',
           size='Count')
    st.write('According to the map it is visible that the Downtown of Chicago has the biggest density of crimes; '
             'That is expected, as city center is a frequent destination for the tourists and business visitors, who is the frequent target of the criminals')

    st.markdown('#')

    st.title('Thesis #2')
    null_hypothesis = f'The Primary Type of crime, do not significantly improve the accuracy of predicting the Arrest outcome compared to a random guess.'
    alternative_hypothesis = f'The Primary Type of crime, significantly improve the accuracy of predicting the Arrest outcome compared to a random guess.'
    run_attribute_analysis(null_h=null_hypothesis, alter_h=alternative_hypothesis, column_x='Primary Type', column_y='Arrest', alpha_score=0.05, convert_data=True)
    df_copy = data_processor.data.copy()
    df_copy = df_copy.rename(columns={'Primary Type': 'PrimaryType'})
    df_copy['Arrest'] = df_copy['Arrest'].astype(int)
    df_copy['PrimaryType'] = df_copy['PrimaryType'].astype('category').cat.codes
    model = smf.logit(formula='Arrest ~ PrimaryType', data=df_copy).fit()
    st.write(model.summary())

    st.markdown('#')

    st.title('Thesis #3')
    st.write('Chicago has a humid continental climat, which means that winters there are pretty rough with daylight reducing down to 9.5 hours a day (according to https://www.ncei.noaa.gov/) '
             'Based on this data, assumption is that most of the crimes are commited when there is less visibility throughout the day, that is - during the winter ')

    data_by_month = data_processor.df_grouped_by_month()
    ml_df = data_by_month.copy()
    ml_df['Month'] = pd.to_datetime(ml_df['Month'])
    ml_df['Month_Num'] = ml_df['Month'].dt.month
    model = smf.glm(formula='Count ~ Month_Num', data=ml_df, family=sm.families.Poisson()).fit()
    st.write(model.summary())

    st.write(r"$\textsf{\Large Crime by Month:}$")
    st.dataframe(data_by_month)
    st.line_chart(data_by_month, x='Month', y=['Count'])
    st.write('Statistical analysis did not reveal any deviations for month-to-month difference between each subsequent month with regard to the crime rate')
    st.markdown('#')

def run_attribute_analysis(null_h, alter_h, column_x, column_y, alpha_score: float, convert_data: bool = True, ml_classifier=RandomForestClassifier, data=None):
    print_large('Null Hypothesis (H0)')
    st.write(null_h)
    print_large('Alternative Hypothesis (H1)')
    st.write(alter_h)
    print_large(f'Running ML classification test with {ml_classifier.__name__} model')
    run_ml_classifier(column_x=column_x, column_y=column_y, alpha_score=alpha_score, model=ml_classifier, convert_data=convert_data, data=data)
    st.markdown('#')


@st.cache_data
def run_ml_classifier(column_x, column_y, alpha_score: float, convert_data: bool = True, test_size=0.3, n_estimators=100, random_state=42, model=RandomForestClassifier, data=None):
    if data is None:
        data = data_processor.data.copy()
    label_encoder_arrest = LabelEncoder()
    data[column_y] = label_encoder_arrest.fit_transform(data[column_y])
    all_columns = [column_y]
    if isinstance(column_x, list):
        all_columns += column_x
    else:
        all_columns.append(column_x)

    data = data[all_columns]
    data = data.dropna(subset=all_columns)

    if convert_data:
        data = pd.get_dummies(data.copy(), columns=[column_x])
    if isinstance(column_x, list):
        X = data[column_x]
    else:
        X = data.drop(columns=[column_y])
    y = data[column_y]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    st.write(f"Trying to find correlation with {model.__name__}")
    st.write(f"{test_size = }")
    st.write(f"{n_estimators = }")
    st.write(f"{random_state = }")
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(x_train, y_train)

    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(x_train, y_train)

    rf_pred = rf_model.predict(x_test)
    dummy_pred = dummy_model.predict(x_test)

    rf_mse = mean_squared_error(y_test, rf_pred)
    dummy_mse = mean_squared_error(y_test, dummy_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    dummy_r2 = r2_score(y_test, dummy_pred)
    st.write(f'Random Forest MSE: {rf_mse}, R2: {rf_r2}')
    st.write(f'Dummy Regressor MSE: {dummy_mse}, R2: {dummy_r2}')

    rf_accuracy = accuracy_score(y_test, rf_pred)
    dummy_accuracy = accuracy_score(y_test, dummy_pred)
    st.write(f'Dummy Classifier Accuracy: {dummy_accuracy:.2%}')
    st.write(f'Random Forest Accuracy: {rf_accuracy:.2%}')

    st.write(f'Random Forest Classification Report:\n')
    report = classification_report(y_test, rf_pred, output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(report))

    st.write(f'Dummy Classification Report:\n')
    report = classification_report(y_test, dummy_pred, output_dict=True)
    st.dataframe(pd.DataFrame.from_dict(report))

    st.write(f'Random Forest Confusion Matrix:\n')
    conf_matrix = confusion_matrix(y_test, rf_pred)
    st.write(conf_matrix)
    st.write(f'Dummy Classifier Confusion Matrix:\n')
    conf_matrix = confusion_matrix(y_test, dummy_pred)
    st.write(conf_matrix)

    k = 5
    rf_scores = cross_val_score(rf_model, X, y, cv=k, scoring='neg_mean_squared_error')
    dummy_scores = cross_val_score(dummy_model, X, y, cv=k, scoring='neg_mean_squared_error')

    st.write(f'Random Forest Mean Accuracy: {np.mean(rf_scores)}')
    st.write(f'Random Forest Accuracy Standard Deviation: {np.std(rf_scores)}')
    st.write(f'Dummy Classifier Mean Accuracy: {np.mean(dummy_scores)}')
    st.write(f'Dummy Classifier Accuracy Standard Deviation: {np.std(dummy_scores)}')

    t_stat, p_value = ttest_ind(rf_scores, dummy_scores, equal_var=False)

    st.write(f'T-statistic: {t_stat}')
    st.write(f'P-value: {p_value}')

    print_large("Random Forest Conclusion:")
    if p_value < alpha_score:
        st.write("Reject the null hypothesis: The Random Forest model performs significantly better than the dummy classifier.")
    else:
        st.write("Fail to reject the null hypothesis: The Random Forest model does not perform significantly better than the dummy classifier.")
    plt.figure(figsize=(10, 6))
    plt.boxplot([rf_scores, dummy_scores], labels=['Random Forest', 'Dummy Classifier'])
    plt.ylabel('Accuracy')
    plt.title('Cross-Validation Accuracy Scores for Random Forest and Dummy Classifier')
    st.pyplot(plt)


def print_large(s: str):
    st.write(r"$\textsf{\Large " + s + r"}$")


if __name__ == '__main__':
    main()
