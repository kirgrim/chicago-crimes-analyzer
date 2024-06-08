import streamlit as st
import altair as alt

from processor import data_processor


def main():
    st.title("Chicago Crimes in 2024")
    data_processor.reset_filters()

    st.markdown('##')

    st.write(r"$\textsf{\Large Use these filters to adjust the displayed data:}$")
    precision = st.slider("Adjust coordinates precision with slider", 1, 5, 2)
    selected_crimes = st.multiselect("Select types of crimes (displays all by default)",
                                     data_processor.get_unique_column_values(column='Primary Type'))
    data_processor.apply_filters(data_filters={'Primary Type': selected_crimes})

    st.markdown('#')

    st.title('Thesis #1')
    st.write('Visitors of the city are likely to be the main target of the criminals')
    st.write(r"$\textsf{\Large Map of Chicago Crimes:}$")
    st.map(data_processor.df_grouped_by_coordinates(precision=precision),
           latitude='Latitude',
           longitude='Longitude',
           size='Count')
    st.write('According to the map it is visible that the Downtown of Chicago has the biggest density of crimes; '
             'That is expected, as city center is a frequent destination for the tourists and business visitors, who is the frequent target of the criminals')

    st.markdown('#')

    st.title('Thesis #2')
    st.write('According to the police reports, theft is the most frequently occurring crime')

    st.markdown('#')

    st.write(r"$\textsf{\Large Crime by Popularity:}$")
    st.bar_chart(data_processor.df_grouped_by_property(column_name='Primary Type'), x='Primary Type', y=['Count'])

    st.write('Chart above illustrates that only in first half of year 2024 there is over 22000 officially registered thefts in Chicago, '
             'this number outperforms any other forms of crimes')

    st.markdown('#')

    st.title('Thesis #3')
    st.write('As is was stated in Thesis #2 - theft is the most frequent crime. Is there a direct relation between the crime nature and frequency of the word occurrence from police descriptions')
    st.pyplot(data_processor.df_build_wordcloud_from_column(column_name='Description'))
    st.write('As is visible from the wordcloud - Theft is in the top of the most frequently occurring words, but not as much frequent as others e.g. BATTERY')

    st.markdown('#')

    st.title('Thesis #4')
    st.write('Chicago has a humid continental climate, which means that winters there are pretty rough with daylight reducing down to 9.5 hours a day (according to https://www.ncei.noaa.gov/);'
             'Based on this data, assumption is that most of the crimes are commited when there is less visibility throughout the day, that is - during the winter ')

    st.markdown('#')

    data_by_month = data_processor.df_grouped_by_month()
    st.write(r"$\textsf{\Large Crime by Month:}$")
    st.dataframe(data_by_month)
    st.line_chart(data_by_month, x='Month', y=['Count'])
    st.write('Thesis is proven to be wrong, seems that there is a wavelike month-to-month difference between each subsequent month')


if __name__ == '__main__':
    main()
