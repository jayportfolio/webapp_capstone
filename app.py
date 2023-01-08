import random

import pandas as pd
import streamlit as st
import pickle

#from functions_20221019B import this_test_data, get_source_dataframe
#from functions_20221109 import this_test_data, get_source_dataframe
from functions_d3__prepare_store_data_20221116 import this_test_data
from functions_b__get_the_data_20221116 import get_source_dataframe

import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

df, X_test, y_test = None, None, None
rand_index = -1

DEFAULT_MODEL = 'Decision Tree'


def main():
    global X_test, y_test, rand_index
    fake_being_in_colab = True

    st.markdown(
        "<h1 style='text-align: center; color: White;background-color:#e84343'>London Property Prices Predictor</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: Black;'>Insert your property parameters here, or choose a random pre-existing property.</h3>",
        unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: Black;'>Sub heading here</h4>",
                unsafe_allow_html=True)

    st.sidebar.header("What is this Project about?")
    st.sidebar.markdown(
        "This is a Web app that would predict the price of a London property based on parameters.")
    st.sidebar.header("Sidebar Options")
    include_nulls = st.sidebar.checkbox('include rows with any nulls ')
    if st.sidebar.button('Purge everything'):
        #st.sidebar.error("I haven't added this functionality yet")
        purge_everything()
        st.sidebar.error("Everything purged")

    available_models = [
        'final_model_KNN_v06',
        'final_model_XG Boost (tree)_v11',
        'final_model_CatBoost_v06',
        'final_model_XG Boost (tree)_v09',
        'final_model_XG Boost (tree)_v06',
        'final_model_Neural Network m15_v09',
        'final_model_Linear Regression (Ridge)_v06'
    ]
    selected_model = st.selectbox('Which model do you want to use?', available_models)

    try:
        model_path = f'models/{selected_model}.pkl'
        model = pickle.load(open(model_path, 'rb'))
    except:
        raise ValueError(f'no model: {model_path}')

    DATA_VERSION = '06'
    DATA_VERSION = selected_model[-2:]

    manual_parameters = st.checkbox('Use manual parameters instead of sample')
    if not manual_parameters:
        X_test, y_test = this_test_data(VERSION=DATA_VERSION, test_data_only=True, IN_COLAB=fake_being_in_colab)

        test_size = len(y_test)

    else:
        lati = st.slider("Input Your latitude", 51.00, 52.00)
        longi = st.slider("Input your longitude", -0.5, 0.3)
        beds = st.slider("Input number of bedrooms", 0, 6)
        baths = st.slider("Input number of bathrooms", 0, 6)

        inputs = [[lati, longi, beds, baths]]

    if st.button('Predict',):

        if not manual_parameters:
            try:
                print('------------- 1 -----------------')
                random_instance_plus = np.loadtxt("random_instance_plus.csv", delimiter=",")
                random_instance_plus = np.loadtxt("random_instance_plus.csv", delimiter=",")
                rand_index = int(random_instance_plus[0])
                expected = random_instance_plus[1]
                inputs = [random_instance_plus[2:]]
            except:
                print('------------- 2 -----------------')
                # raise ValueError()
                rand_index = random.randint(0, test_size - 1)
                inputs = [X_test[rand_index]]
                random_instance = inputs
                expected = y_test[rand_index]
                print('try to save random_instance', random_instance)
                np.savetxt("random_instance.csv", random_instance, delimiter=",")
                print('finished save random_instance')
                random_instance_plus = [rand_index, expected]
                print('0 random_instance', random_instance, type(random_instance))
                print('1 random_instance_plus', random_instance_plus, type(random_instance_plus))
                #random_instance_plus.extend(random_instance)
                for each2 in random_instance:
                    for each in each2:
                        random_instance_plus.append(each)
                print('2 random_instance_plus', random_instance_plus, type(random_instance_plus))
                print("random_instance_plus:", random_instance_plus)
                print(random_instance_plus)
                print(type(random_instance_plus))
                print('try to save random_instance_plus', random_instance_plus)
                np.savetxt("random_instance_plus.csv", random_instance_plus, delimiter=",")
                print('finished save random_instance_plus')

            st.text(f'Actual value of property {rand_index}: {expected}')

        print("inputs:", inputs)

        try:
            result = model.predict(inputs)
        except:
            purge_everything()
            X_train, X_test, y_train, y_test = this_test_data(VERSION=DATA_VERSION, IN_COLAB=fake_being_in_colab)
            result = model.predict(inputs)

        updated_res = result.flatten().astype(float)
        st.success('The predicted price for this property is £ {}'.format(updated_res[0]))
        st.warning('The actual price for this property is £ {}'.format(expected))


    if st.checkbox('Get multiple predictions (entire test set)'):
        X_train, X_test, y_train, y_test = this_test_data(VERSION=DATA_VERSION)
        acc = model.score(X_test, y_test)
        st.write('Accuracy of test set: ', acc)

        multiple_predictions = np.vstack((y_test.flatten(), model.predict(X_test).flatten())).T
        multiple_predictions_df = pd.DataFrame(multiple_predictions, columns=['Actual Price','Predicted Price'])
        st.write(multiple_predictions_df)
        print(type(multiple_predictions_df ))

    if not manual_parameters:
        if st.button('Get a different random property!'):
            rand_index = random.randint(0, test_size - 1)
            inputs = [X_test[rand_index]]

            random_instance = inputs
            np.savetxt("random_instance.csv", random_instance, delimiter=",")
            st.text(f'sample variables ({rand_index}): {inputs[0]}')
            st.text(f'Expected prediction: {y_test[rand_index]}')

            expected = y_test[rand_index]
            np.savetxt("random_instance.csv", random_instance, delimiter=",")
            random_instance_plus = [rand_index, expected]
            random_instance_plus.extend(random_instance[0])
            print("random_instance_plus:", random_instance_plus)
            np.savetxt("random_instance_plus.csv", [random_instance_plus], delimiter=",")

    #df = get_source_dataframe(IN_COLAB=False, VERSION=DATA_VERSION, folder_prefix='')
    if st.checkbox('Show the underlying dataframe'):
        #df, df_type = get_source_dataframe(IN_COLAB=False, VERSION=DATA_VERSION, folder_prefix='')
        df, df_type = get_source_dataframe(cloud_run=fake_being_in_colab, VERSION=DATA_VERSION, folder_prefix='')
        print("claiming to be colab so I can use the cloud version of data and save space")
        st.write(df)


def purge_everything():
    # importing the os Library
    import os
    for deletable_file in [
        'train_test/X_test.csv', 'train_test/X_test_no_nulls.csv', 'train_test/_train.csv',
        'train_test/X_train_no_nulls.csv',
        'train_test/y_test.csv', 'train_test/y_test_no_nulls.csv', 'train_test/y_train.csv',
        'train_test/y_train_no_nulls.csv',
        'models/model_Decision Tree.pkl',
        'models/model_Deep Neural Network.pkl',
        'models/model_HistGradientBoostingRegressor.pkl',
        'models/model_Linear Regression.pkl',
        'models/model_Linear Regression (Keras).pkl',
        'random_instance.csv',
        'random_instance_plus.csv',
        # functions.FINAL_RECENT_FILE,
        # functions.FINAL_RECENT_FILE_SAMPLE,
    ]:
        # checking if file exist or not
        if (os.path.isfile(deletable_file)):

            # os.remove() function to remove the file
            os.remove(deletable_file)

            # Printing the confirmation message of deletion
            print("File Deleted successfully:", deletable_file)
        else:
            print("File does not exist:", deletable_file)
        # Showing the message instead of throwig an error


if __name__ == '__main__':
    main()
