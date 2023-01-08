import ast
import math

import numpy as np
import pandas as pd


def tidy_dataset(df, version: int) -> pd.DataFrame:
    if version >= 2:
        df = df[df['sharedOwnership'] == False]

    return df

def add_supplements(property_dataset, version):
    version_number = int(version)

    property_dataset['Price'] = pd.to_numeric(property_dataset['Price'], 'coerce').dropna().astype(int)

    # do any necessary renames, and some preliminary feature engineering
    try:
        property_dataset = property_dataset.rename(index=str, columns={"Station_Prox": "distance_to_any_train"})
    except:
        pass

    try:
        property_dataset['borough'] = property_dataset["borough"].str.extract("\('(.+)',")
    except:
        pass

    def simplify(array_string):
        try:
            array = array_string.split("/")  # a list of strings
            return array[0]
        except:
            pass

    try:
        property_dataset['propertyType'] = property_dataset['analyticsProperty.propertyType'].apply(simplify)
    except:
        pass

    try:
        property_dataset['coarse_compass_direction'] = property_dataset["address.outcode"].str.extract("([a-zA-Z]+)")
    except:
        pass

    try:
        property_dataset['sq_ft'] = property_dataset["size"].str.extract("(\d*) sq. ft.")
    except:
        pass

    property_dataset = property_dataset[(property_dataset['Price'] >= 100000) & (property_dataset['Price'] <= 600000)]

    if version_number > 3:
        property_dataset['sharedOwnership'] = (
                (property_dataset['sharedOwnership.sharedOwnership'] == True) |
                (property_dataset['analyticsProperty.priceQualifier'] == 'Shared ownership') |
                (property_dataset['keyFeatures'].str.contains('shared ownership'))
        )

        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership'])
        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
                property_dataset['sharedOwnership.sharedOwnership'] == 1)
        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
                property_dataset['analyticsProperty.priceQualifier'] == 'Shared ownership')

        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
            property_dataset['keyFeatures'].str.contains('shared ownership'))
        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
                (property_dataset['keyFeatures'].str.contains('share')) & (
            property_dataset['keyFeatures'].str.contains('%')))

        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
            property_dataset['text.description'].str.contains('shared ownership'))
        property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
                (property_dataset['text.description'].str.contains('share')) & (
            property_dataset['text.description'].str.contains('%')))

    if version_number > 4:
        property_dataset['keyFeatures'] = property_dataset['keyFeatures'].str.lower()
        property_dataset['text.description'] = property_dataset['text.description'].str.lower()


    # property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (property_dataset['sharedownership_in_description'])

    try:
        property_dataset['sharePercentage'] = property_dataset.apply(share_percentage, axis=1)
    except:
        pass


    if version_number > 3:
        property_dataset['nearestStation'] = property_dataset['nearestStations'].apply(stations, args=['any'])
        property_dataset['nearestTram'] = property_dataset['nearestStations'].apply(stations, args=['TRAM'])
        property_dataset['nearestUnderground'] = property_dataset['nearestStations'].apply(stations,
                                                                                           args=['LONDON_UNDERGROUND'])
        property_dataset['nearestOverground'] = property_dataset['nearestStations'].apply(stations,
                                                                                          args=['overground'])

    if version_number in [9, 10, 11]:
        dff, aggregate_features = aggregate_keyFeatures(property_dataset['keyFeatures'])
        property_dataset['reduced_features'] = property_dataset['keyFeatures'].apply(reduce_keyFeatures, args=[dff.index,
                                                                                                               get_exploding_features(
                                                                                                                   version_number)])

    return property_dataset


def share_percentage(df_row):
    #        print(df_row)
    if df_row['sharedOwnership.sharedOwnership']:
        if type(df_row['sharedOwnership.ownershipPercentage']) in [int, float] and not math.isnan(
                df_row['sharedOwnership.ownershipPercentage']):
            return df_row['sharedOwnership.ownershipPercentage']
        else:
            return None
    else:
        return 100


def stations(station_list_string, requested_type):
    # print('stations')
    # pass
    # station_list = json.loads(station_list_string)
    import ast
    station_list = ast.literal_eval(station_list_string)

    # print('---')
    # print(station_list)

    # NATIONAL_TRAIN
    # LIGHT_RAILWAY
    # TRAM
    for station in station_list:

        if station['types'] not in [
            ['NATIONAL_TRAIN'],
            ['LONDON_UNDERGROUND'],
            ['LIGHT_RAILWAY'],
            ['LONDON_OVERGROUND'],
            ['TRAM'],
            ['CABLE_CAR'],
            ['LONDON_UNDERGROUND', 'LIGHT_RAILWAY'],
            ['LIGHT_RAILWAY', 'LONDON_OVERGROUND'],
            ['LONDON_UNDERGROUND', 'LONDON_OVERGROUND'],
            ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LONDON_OVERGROUND'],
            ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND'],
            ['NATIONAL_TRAIN', 'LIGHT_RAILWAY'],
            ['NATIONAL_TRAIN', 'TRAM'],
            ['NATIONAL_TRAIN', 'LONDON_OVERGROUND'],
            ['NATIONAL_TRAIN', 'TRAM', 'LONDON_OVERGROUND'],
            ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LIGHT_RAILWAY'],
            ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'TRAM'],
            ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LIGHT_RAILWAY', 'LONDON_OVERGROUND'],
        ]:
            print(f"WARNING: Station type not found: {station['types']}: {station}")

        if requested_type == 'any':
            # print(station)
            return station['distance']
        elif requested_type in station['types']:
            return station['distance']
        elif requested_type == "overground" and (
                'NATIONAL_TRAIN' in station['types'] or 'LONDON_OVERGROUND' in station[
            'types'] or 'LIGHT_RAILWAY' in station['types']):
            return station['distance']
        elif requested_type == "underground combined" and 'LONDON_UNDERGROUND' in station['types'] and len(
                station['types']) > 1:
            return station['distance']
        else:
            pass

    return 99

def aggregate_keyFeatures(key_features_column):
    from string import punctuation

    aggregated_features = []

    def clean(text):
        clean_each = [x.replace('*', '') for x in text]
        clean_each = [x.lstrip('-') for x in clean_each]
        clean_each = [x.replace('\t', '') for x in clean_each]
        clean_each = [x.lstrip('-') for x in clean_each]
        clean_each = [x.lstrip('-') for x in clean_each]
        clean_each = [x.lstrip('•') for x in clean_each]
        return clean_each

    for each_string in key_features_column:
        each = ast.literal_eval(each_string)
        # print("each\n", each)

        clean_each = clean(each)

        aggregated_features.extend(clean_each)
    ar_unique, i = np.unique(aggregated_features, return_counts=True)
    print(ar_unique, i)
    print(len(ar_unique), len(i))
    comb = np.vstack((ar_unique, i))
    dff = pd.DataFrame(comb.T, columns=['feature', 'occurrences'])
    dff['occurrences'] = pd.to_numeric(dff['occurrences'], 'coerce').dropna().astype(int)

    dff.sort_values('occurrences', ascending=False, inplace=True)
    dff.set_index('feature', inplace=True)

    return dff, aggregated_features


def reduce_keyFeatures(key_features_string, full_features, max_features):
    features_list = ast.literal_eval(key_features_string)

    allowed_features = full_features[:max_features]

    filtered_features_list = [x for x in features_list if x in allowed_features]
    if len(filtered_features_list) > 0:
        if False:
            print(f"{features_list}\n==>   {filtered_features_list}")

    return filtered_features_list


def get_exploding_features(version_number):
    if version_number < 9:
        EXPLODING_FEATURES = 0
    elif version_number == 9:
        EXPLODING_FEATURES = 10
    elif version_number == 10 or version_number == 11:
        EXPLODING_FEATURES = 50
    else:
        raise ValueError(f'missing parameter: version_number')

    return EXPLODING_FEATURES

