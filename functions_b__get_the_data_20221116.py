import pandas as pd

# csv_directory = "final_split"
# csv_directory = "quick_split"
csv_directory = "DDD"

FINAL_BASIC_FILE = "data/DDD/listings_data_basic_XXX.csv"
FINAL_ENRICHED_FILE = "data/DDD/listings_data_enriched_XXX.csv"
FINAL_JSON_MODEL_FILE = "data/DDD/listings_data_jsonmodel_XXX.csv"
FINAL_JSON_META_FILE = "data/DDD/listings_data_jsonmeta_XXX.csv"
FINAL_RECENT_FILE = "data/source/df_listings.csv"
FINAL_RECENT_FILE_SAMPLE = "data/sample/df_listings_sample.csv"


def set_csv_directory(update_csv_directory):
    global csv_directory
    csv_directory = update_csv_directory


def get_source_dataframe(cloud_run, VERSION, row_limit=None, folder_prefix='../../../'):

    filename = f'df_listings_v{VERSION}.csv'
    remote_pathname = f'https://raw.githubusercontent.com/jayportfolio/capstone_streamlit/main/data/final/{filename}'
    df_pathname_raw = folder_prefix + f'data/source/{filename}'
    df_pathname_tidy = folder_prefix + f'data/final/{filename}'

    if cloud_run:
        df = pd.read_csv(remote_pathname, on_bad_lines='error', index_col=0)
        retrieval_type = 'tidy'
        print('loaded data from', folder_prefix + remote_pathname)
    else:
        df = pd.read_csv(df_pathname_tidy, on_bad_lines='error', index_col=0)
        retrieval_type = 'tidy'
        print('loaded data from', df_pathname_tidy)

    if row_limit and row_limit > 0:
        df = df[:row_limit]
    return df, retrieval_type


def get_combined_dataset(HOW, early_duplicates, row_limit=None, verbose=False, folder_prefix=''):
    df_list = get_df(FINAL_BASIC_FILE, folder_prefix)
    # df_indiv = get_df(FINAL_ENRICHED_FILE, testing)
    df_indiv = get_df(FINAL_ENRICHED_FILE, folder_prefix)
    df_meta = get_df(FINAL_JSON_META_FILE, folder_prefix)
    # df_json1 = get_df(LISTING_JSON_MODEL_FILE, on_bad_lines='warn')  # EDIT 29-06-2022: There are bid listings and regular listings. I scrape them seporately and join them here.
    # df_json = get_df(LISTING_JSON_MODEL_FILE)
    df_json = get_df(FINAL_JSON_MODEL_FILE, folder_prefix)

    df_meta['id_copy'] = df_meta['id_copy'].astype(int)
    df_json['id'] = df_json['id'].astype(int)
    df_list.set_index(['ids'], inplace=True)
    df_indiv.set_index(['ids'], inplace=True)
    df_meta.set_index(['id_copy'], inplace=True)
    df_json.set_index(['id'], inplace=True)
    # df_age.set_index(['ids'], inplace=True)

    if HOW == 'no_indexes':
        df_original = df_list \
            .merge(df_json, left_on='ids', right_on='id', how=HOW, suffixes=('', '_model')) \
            .merge(df_meta, left_on='ids', right_on='id_copy', how=HOW, suffixes=('', '_meta'))
        # .merge(df_indiv, on='ids', how='inner', suffixes=('', '_listing')) \
    elif HOW == 'listings_only':
        df_original = df_list
    elif HOW == 'left':  # https://www.statology.org/pandas-merge-on-index/
        df_original = df_list \
            .join(df_json, how=HOW, lsuffix='', rsuffix='_model') \
            .join(df_meta, how=HOW, lsuffix='', rsuffix='_meta')  # \
        # .join(df_indiv, on='ids', how='inner', lsuffix='', rsuffix='_listing') \
        # .join(df_age, how=HOW, lsuffix='', rsuffix='_age')
    elif HOW == 'inner2':  # https://www.statology.org/pandas-merge-on-index/
        df_original = df_list \
            .join(df_json, how='inner', lsuffix='', rsuffix='_model') \
            .join(df_meta, how='inner', lsuffix='', rsuffix='_meta') \
            .join(df_indiv, how='inner', lsuffix='', rsuffix='_listing')
        # .join(df_age, how=HOW, lsuffix='', rsuffix='_age')
    elif HOW == 'inner':  # https://www.statology.org/pandas-merge-on-index/
        df_original = pd.merge(
            pd.merge(pd.merge(df_list, df_indiv, left_index=True, right_index=True, suffixes=('', '_listing'))
                     , df_json, left_index=True, right_index=True, suffixes=('', '_model'))
            , df_meta, left_index=True, right_index=True, suffixes=('', '_meta'))
    else:
        raise LookupError(f"no HOW parameter called {HOW}")

    # df_original.to_csv(vv.QUICK_COMBINED_FILE, mode='w', encoding="utf-8", index=True, header=True)
    del df_list
    # del df_indiv
    del df_json
    del df_meta

    df_original.iloc[:20].to_csv(folder_prefix + 'data/source/df_source_full_sample.csv')
    df_original.to_csv(folder_prefix + 'data/source/df_source_full.csv')

    if row_limit and row_limit > 0:
        # return df_original[:row_limit]
        return df_original.sample(n=row_limit)

    if early_duplicates:
        df_original = df_original[~df_original.index.duplicated(keep='last')]

    return df_original


def get_df(file, folder_prefix=''):
    df_array = []
    for n in range(10):
        prefix = '00' if n <= 10 else '0'

        filename = folder_prefix + file.replace('XXX', prefix + str(n)).replace('DDD', csv_directory)

        from os.path import exists
        file_exists = exists(filename)

        if file_exists:
            if n == 0:
                merged_df = pd.read_csv(filename, on_bad_lines='skip')
            else:
                # each_df = pd.read_csv(filename, on_bad_lines='skip')
                # df_array.append(each_df)
                merged_df = pd.concat([merged_df, pd.read_csv(filename, on_bad_lines='skip')])
                # print(n)

            # print(f'error on {file} at split {prefix}')
        else:
            if n == 1:
                raise LookupError("didn't find ANY matching files! filename: " + filename)
            # print(f'{n + 1} splits for {file}')
            break

    return merged_df
