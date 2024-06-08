import os
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from wordcloud import WordCloud


class ChicagoDataProcessor:

    SOURCE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'Crimes_2024.csv')
    DATA_NON_EMPTY_COLUMNS = (
        'Date',
        'Block',
        'Primary Type',
        'Location Description',
        'Arrest',
        'Domestic',
        'District',
    )

    CONSIDERED_COLUMNS = (
        'Date',
        'Block',
        'Primary Type',
        'Description',
        'Location Description',
        'Arrest',
        'Domestic',
        'District',
        'Latitude',
        'Longitude',
    )

    def __init__(self):
        self.initial_data = self._clean_data(pd.read_csv(self.SOURCE_DATA_PATH))
        self.data = self.initial_data.copy(deep=True)

    def apply_filters(self, data_filters: dict[str, list[Any]]):
        if data_filters:
            for k, v in data_filters.items():
                if v:
                    self.data = self.data[self.data[k].isin(v)]

    def reset_filters(self):
        self.data = self.initial_data.copy(deep=True)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError('Obtained Empty DataFrame')
        df = df[list(self.CONSIDERED_COLUMNS)]
        df.dropna(subset=list(self.DATA_NON_EMPTY_COLUMNS), inplace=True)
        return df

    def get_unique_column_values(self, column: str) -> list[Any]:
        return self.data[column].unique()

    def df_grouped_by_coordinates(self, precision: int = 3):

        data_valid_coordinates = self.data.copy()
        data_valid_coordinates.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        df = data_valid_coordinates.copy()

        def round_coordinates(lat, lon):
            return round(lat, precision), round(lon, precision)

        df['lat_group'], df['lon_group'] = zip(*df.apply(lambda row: round_coordinates(row['Latitude'], row['Longitude']), axis=1))
        grouped = df.groupby(['lat_group', 'lon_group']).size().reset_index(name='count')

        return pd.DataFrame({
            'Latitude': grouped['lat_group'],
            'Longitude': grouped['lon_group'],
            'Count': grouped['count']
        })

    def df_grouped_by_property(self, column_name: str) -> pd.DataFrame:
        grouped = self.data.groupby([column_name]).size().reset_index(name='count')
        return pd.DataFrame({
            column_name: grouped[column_name],
            'Count': grouped['count']
        })

    def df_grouped_by_month(self):
        df = self.data.copy()
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S %p')
        df['Month'] = df['Date'].dt.to_period('M').astype(str)
        grouped = df.groupby('Month').size().reset_index(name='count')
        print(grouped.head())
        return pd.DataFrame({
            'Month': grouped['Month'],
            'Count': grouped['count']
        })

    def df_build_wordcloud_from_column(self, column_name: str):
        cloud_text = ' '.join(self.data[column_name])
        cloud_text = cloud_text.upper()
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cloud_text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig


data_processor = ChicagoDataProcessor()
