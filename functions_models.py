from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from functions import this_test_data


def build_model(algorithm, drop_nulls=False):
    X_train, X_test, y_train, y_test = this_test_data(drop_nulls=drop_nulls)

    if algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

    elif algorithm == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif algorithm == 'HistGradientBoostingRegressor':
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)

    elif algorithm == 'Deep Neural Network':

        import tensorflow as tf

        from tensorflow import keras
        from tensorflow.keras import layers

        print(tf.__version__)

        def build_and_compile_model(norm):
            model = keras.Sequential([
                norm,
                layers.Dense(132, activation='relu'),
                layers.Dense(132, activation='relu'),
                #layers.Dense(400, activation='relu'),
                #layers.Dense(400, activation='relu'),
                #layers.Dense(400, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(loss='mean_absolute_error',
                          optimizer=tf.keras.optimizers.Adam(0.001))
            return model

        normalizer = tf.keras.layers.Normalization(axis=-1)

        dnn_model = build_and_compile_model(normalizer)
        # print(dnn_model.summary())

        # % % time
        history = dnn_model.fit(
            X_train,  # train_features,
            y_train,  # train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        # def plot_loss(history):
        #     import matplotlib.pyplot as plt
        #     plt.plot(history.history['loss'], label='loss')
        #     plt.plot(history.history['val_loss'], label='val_loss')
        #     plt.ylim([0, 10])
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Error [MPG]')
        #     plt.legend()
        #     plt.grid(True)
        #
        # plot_loss(history)

        # test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
        print(dnn_model.evaluate(X_test, y_test, verbose=0))

        model = dnn_model

    elif algorithm == 'Linear Regression (Keras)':
        import tensorflow as tf

        from tensorflow import keras
        from tensorflow.keras import layers

        print(tf.__version__)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        linear_model = tf.keras.Sequential([
            normalizer,
            layers.Dense(units=1)
        ])

        linear_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        # %%time
        history = linear_model.fit(
            X_train,  # train_features,
            y_train,  # train_labels,
            epochs=100,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2)

        # def plot_loss(history):
        #     import matplotlib.pyplot as plt
        #     plt.plot(history.history['loss'], label='loss')
        #     plt.plot(history.history['val_loss'], label='val_loss')
        #     plt.ylim([0, 10])
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Error [MPG]')
        #     plt.legend()
        #     plt.grid(True)
        #
        # plot_loss(history)

        model = linear_model

    elif algorithm == 'Linear Regression (Keras)':
        from tensorflow_estimator.python.estimator.canned.linear import LinearRegressor

        model = LinearRegressor()
        model.fit(X_train, y_train)
    else:
        raise ValueError(algorithm)

    return model
