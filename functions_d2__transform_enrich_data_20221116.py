import ast

import numpy as np
import pandas as pd
import pandas as pd

import math


def preprocess(df, version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number == 2:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)
        df = df[(df['location.longitude'] <= 10)]
        df = df[(df['bedrooms'] <= 10)]
        df = df[df['bathrooms'] <= 5]
        df = df[(df['nearestStation'] <= 20)]

    elif version_number == 3 or version_number == 4:
        df = df[(df['longitude_deviation'] <= 1)]
        df = df[(df['bedrooms'] <= 10)]
        df = df[df['bathrooms'] <= 5]
        df = df[(df['nearestStation'] <= 20)]

    elif version_number == 5:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)
        df = df[(df['location.longitude'] <= 10)]

        df = df[(df['longitude_deviation'] <= 1)]
        df = df[(df['bedrooms'] <= 10)]
        df = df[df['bathrooms'] <= 5]
        df = df[(df['nearestStation'] <= 20)]

    elif version_number <= 12:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)

        df = df[(df['bedrooms'] <= 7)]
        df = df[df['bathrooms'] <= 5]

        df = df[(df['nearestStation'] <= 7.5)]

        df = df[(df['location.longitude'] <= 1)]
        df = df[(df['longitude_deviation'] <= 1)]

    else:
        raise ValueError(f'no columns data available for version {version_number}')

    return df


def feature_engineer(df, version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number >= 3:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)
        df['location.longitude'] = pd.to_numeric(df['location.longitude'], 'coerce').dropna().astype(float)

        average_latitude0 = df['location.latitude'].mean()
        average_longitude0 = df['location.longitude'].mean()

        average_latitude1 = df['location.latitude'].median()
        average_longitude1 = df['location.longitude'].median()

        average_latitude2 = 51.4626624
        average_longitude2 = -0.0651048

        df['latitude_deviation'] = abs(df['location.latitude'] - average_latitude1)
        df['longitude_deviation'] = abs(df['location.longitude'] - average_longitude1)

        df['latitude_deviation2'] = abs(df['location.latitude'] - average_latitude2)
        df['longitude_deviation2'] = abs(df['location.longitude'] - average_longitude2)

    if version_number in [9, 10, 11, 12]:
        exploded_features_df = (
            df['reduced_features'].explode()
            .str.get_dummies(',').sum(level=0).add_prefix('feature__')
        )
        df = df.drop('reduced_features', 1).join(exploded_features_df)

    if version_number in [11, 12]:
        dailmail = ['garden', 'central heating', 'parking', 'off road', 'shower', 'cavity wall insulation',
                    'wall insulation', 'insulation', 'insulat', 'dining room', 'garage', 'en-suite', 'en suite']
        common_knowledge = ['penthouse', 'balcony']
        ideal_home = ['double-glazing', 'double glazing', 'off-road parking', 'security', 'patio', 'underfloor heating',
                      'marble']
        discarded = ['signal', 'secure doors', 'secure door', 'outdoor lighting', 'bathtub', 'neighbours', ]

        keywords = []
        keywords.extend(dailmail)
        keywords.extend(common_knowledge)
        keywords.extend(ideal_home)

        import re
        spice_df = pd.DataFrame(dict(('feature__2__' + spice, df.keyFeatures.str.contains(spice, re.IGNORECASE))
                                     for spice in keywords))
        df = df.merge(spice_df, how='outer', left_index=True, right_index=True)

    return df


