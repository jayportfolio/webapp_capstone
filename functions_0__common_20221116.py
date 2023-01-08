import pandas as pd

def test_module():
    if True:
        pass
    elif False:
        trial_df = pd.read_csv('data/final/df_listings_v09.csv')
        feature_engineer(trial_df, version=3)
    else:
        trial_df = pd.read_csv('data/source/df_listings_v09.csv')
        # add_supplements(trial_df)
        dff, aggregate_features = aggregate_keyFeatures(trial_df['keyFeatures'])
        trial_df['reduced_features'] = trial_df['keyFeatures'].apply(reduce_keyFeatures,
                                                                     args=[dff.index,
                                                                           get_exploding_features(
                                                                               version_number=9)])
        trial_df = feature_engineer(trial_df, version=10)

        print(trial_df.head(5))

def get_columns(version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number == 2:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude']
        categories = ['tenure.tenureType']
        custom, wildcard = [], []

    elif version_number == 3 or version_number == 4:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom, wildcard = [], []

    elif version_number <= 6:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom, wildcard = [], []

    elif version_number == 7:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom = ['listingHistory.listingUpdateReason']
        wildcard = []
    elif version_number == 8:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation', 'keyFeatures']
        categories = ['tenure.tenureType']
        custom = ['listingHistory.listingUpdateReason']
        wildcard = []
    elif version_number in [9, 10, 11, 12]:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom = []  # ['reduced_features']
        wildcard = ['feature__']

    else:
        raise ValueError(f'no columns data available for version {version}')

    columns = []
    columns.extend(booleans)
    columns.extend(floats)
    columns.extend(categories)
    columns.extend(custom)
    # columns.extend(wildcard)

    return (columns, booleans, floats, categories, custom, wildcard)


if __name__ == '__main__':
    test_module()
