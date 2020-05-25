

# train on all data
# predict for the upcoming week
import os
def do_job(data):
    final_stock_df, stock_pairs, job_id = data
    all_prediction_df = []
    print('inside of do_job')
    import os
    print("Executing our Task on Process {}".format(os.getpid()))

    def prediction_for_upcoming_week(final_stock_df, pairs_of_stocks,  job_id=None, print_idx=1, n_day_sequences=14, 
                                     start_date_training_data='2018-01-01', n_validation_sequences=50, input_batch_size=128, 
                                     input_verbose=1, n_components=100):
        """
        The main entrypoint for training an LSTM network on stock predictions

        :param final_stock_df: The list of stock pairs with correlations over different time ranges, volume
        :para pairs_of_stocks: The list of stock pairs
        :param print_idx: The number of iterations to pass before printing out progress
        :param n_day_sequences: The number of sequences to pass to the LSTM (i.e. the number of days)
        :param start_date_training_data: Filter for data before thie date to train on
        :param n_validation_sequences: Number of sequences to validate on. Should be >= 50
        :param input_batch_size: size of the batches for training NN
        :parm input_verbose: if 1, print out everything otherwise, don't
        """
        print('inside function')
        import datetime
        import pandas as pd
        import glob
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from tqdm import tqdm
        import time
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.decomposition import PCA
        import seaborn as sns
        import holidays
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Bidirectional, BatchNormalization, TimeDistributed, Conv2D, Flatten, MaxPooling2D
        from tensorflow.keras.callbacks import EarlyStopping, Callback
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.constraints import max_norm

        # prepare the data for LSTM model
        def split_sequences(sequences, n_steps, y_col='pg_so_close_corr_rolling_7_days', 
                            start_idx=0, n_val=50, print_idx=100, input_verbose=1,     n_pca_components=100): #2200
            """
            sequences = input_data
            n_steps = n_days of data to give at a time

            only works for the currently set y_col
            """
            if y_col not in sequences.columns:
                raise ValueError('This y col does not exist in this df')

            X, y = list(), list()
            X_val, y_val = list(), list()

            n_sequences = len(sequences)
            print('n_sequences', n_sequences)

            for i in range(start_idx, n_sequences):
                if i == start_idx and input_verbose == 1:
                    print(f"Training idx start at {i}")
                if (i % print_idx == 0) and (i != 0) and input_verbose==1:
                    print(f"Pct finished = {i/n_sequences}")

                # find the end of this pattern
                end_ix = i + n_steps 
                total_end_ix = end_ix + n_val
                # check if we are beyond the dataset
                if (total_end_ix) > n_sequences:
                    print(f"Training idx end at {end_ix}")
                    print('Total idx checked', total_end_ix)
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = np.array(sequences.loc[:, sequences.columns != f"{y_col}"][i:end_ix]), np.array(
                    sequences.loc[:, sequences.columns == f"{y_col}"].shift(-7).fillna(method='ffill').iloc[end_ix-1])


                X.append(seq_x)
                y.append(seq_y)

            val_start_idx = start_idx + n_sequences - (start_idx  + n_val -2)
            for i in range(val_start_idx, n_sequences):
                if i == val_start_idx and input_verbose==1:
                    print(f"Val idx start at {val_start_idx}")
                if (i % print_idx == 0) and i != 0 and input_verbose==1:
                    print(f"Pct finished for val sequences = {i/n_sequences}")
                # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the dataset
                if end_ix > len(sequences) and input_verbose==1:
                    print(f"Val idx end at {end_ix}")
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = np.array(sequences.loc[:, sequences.columns != f"{y_col}"][i:end_ix]), np.array(
                    sequences.loc[:, sequences.columns == f"{y_col}"].shift(-7).fillna(method='ffill').iloc[end_ix-1])


                X_val.append(seq_x)
                y_val.append(seq_y)



            X, y, X_val, y_val = np.array(X), np.array(y), np.array(X_val), np.array(y_val)

            # errors for standard scaler
            X = np.nan_to_num(X.astype(np.float32)) # converting to float 32 throws some infinity errors
            X_val = np.nan_to_num(X_val.astype(np.float32)) # converting to float 32 throws some infinity errors  


            scalers = {}
            for i in range(X.shape[1]):
                scalers[i] = StandardScaler()
                X[:, i, :] = scalers[i].fit_transform(X[:, i, :]) 

            pca_scalers = {}

            new_X = np.zeros((X.shape[0], X.shape[1], n_pca_components))
            for i in range(X.shape[1]):
                pca_scalers[i] = PCA(n_components=n_pca_components) # ~80%
                new_X[:, i, :] = pca_scalers[i].fit_transform(X[:, i, :]) 


            for i in range(X_val.shape[1]):
                X_val[:, i, :] = scalers[i].transform(X_val[:, i, :]) 


            new_X_val = np.zeros((X_val.shape[0], X_val.shape[1], n_pca_components))
            for i in range(X_val.shape[1]):
                new_X_val[:, i, :] = pca_scalers[i].transform(X_val[:, i, :]) 

           # need  to do this again as standard scaler may have nans
            X = np.nan_to_num(X.astype(np.float32)) # converting to float 32 throws some infinity errors
            X_val = np.nan_to_num(X_val.astype(np.float32)) # converting to float 32 throws some infinity errors 
            print('X val shape', X_val.shape)



            return new_X, y, new_X_val, y_val, scalers, pca_scalers






        def build_keras_model(n_steps, n_features, n_units=100, dropout_pct=0.05, n_layers = 1):
            model = Sequential()


            # define CNN model
        #     model.add(TimeDistributed(Conv2D(n_units, kernel_and_pool_size))
        #     model.add(TimeDistributed(MaxPooling2D(pool_size=kernel_and_pool_size))
        #     model.add(TimeDistributed(Flatten()))


            model.add(LSTM(n_units, activation='relu', dropout=dropout_pct, return_sequences=True, input_shape=(n_steps, n_features)))
            model.add(BatchNormalization())
            for _ in range(n_layers):
                model.add(LSTM(n_units, activation='relu', dropout=dropout_pct, return_sequences=True))
                model.add(BatchNormalization())
            model.add(LSTM(n_units, activation='relu', dropout=dropout_pct))
            model.add(BatchNormalization())
            model.add(Dense(n_units))
            model.add(Dense(int(n_units/2)))
            model.add(Dense(1))
            #Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
            #LR = 0.0001
            #clipnorm=1., clipvalue=0.5
            model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='mse', metrics=['mse'])
            return model


        # validation needs to be 40 or index error
        final_stock_df = final_stock_df.dropna()
        final_stock_df = final_stock_df.sort_values(by='date')
        # add this to predictions
        stock_to_industry = pd.read_csv('../data/Industries stock list - all.csv')
        stock_to_industry.symbol = [i.lower() for i in stock_to_industry.symbol]

        final_stock_df = final_stock_df.dropna()
        most_recent_date = final_stock_df.index.max()

        prediction_end = most_recent_date + datetime.timedelta(7)



        test_df = final_stock_df.iloc[-n_day_sequences:, :]



        n_days_corr_predictions = 7


        pct_change_corr = []
        predicted_corr = []
        last_corr_for_prediction_day = []
        pred_dates = []
        first_stock_industries = []
        second_stock_industries = []

        first_model = True

        start = time.time()
        total_n = len(pairs_of_stocks)

        for idx,stock_pairing in enumerate(pairs_of_stocks):
            if idx % print_idx == 0 and input_verbose ==1 :
                print('----------')
                print(f"Stock pairing = {stock_pairing}")
                print(f"Pct finished = {idx/total_n}")
            first_stock_name, second_stock_name = stock_pairing.split('_')

            first_stock_industries.append(stock_to_industry[stock_to_industry.symbol == first_stock_name].industry.values[0])
            second_stock_industries.append(stock_to_industry[stock_to_industry.symbol == second_stock_name].industry.values[0])



            pred_col_name = f"{stock_pairing}_close_corr_rolling_{n_days_corr_predictions}_days"

            # remove the current 7-day corr for this stock
            # for 7 take rolling 7 days corr to the present day to predict off of

            ## TRAINING AND TESTING DATA
            X,y, X_val, y_val, scalers, pca_scalers = split_sequences(
                final_stock_df[final_stock_df.index >= f"{start_date_training_data}"],
                n_day_sequences, start_idx=0, input_verbose=input_verbose,
                n_val=n_validation_sequences, y_col=f"{pred_col_name}"
            ) # 30 steps
            print('finished splitting sequences')

            train_X, train_y = final_stock_df.loc[:, final_stock_df.columns != f"{pred_col_name}"],  final_stock_df[f"{pred_col_name}"].shift(-7).fillna(method='ffill') 
                                                               # get corr from 7 days in the future
            test_X, test_y = np.array(test_df.loc[:, test_df.columns != f"{pred_col_name}"]),  test_df[f"{pred_col_name}"]
            test_X = test_X.reshape(1, test_X.shape[0], test_X.shape[1])
            test_X = np.nan_to_num(test_X.astype(np.float32))
            print('finihsedd text_x')

            for i in range(test_X.shape[1]):
                test_X[:, i, :] = scalers[i].transform(test_X[:, i, :]) 
            test_X = np.nan_to_num(test_X.astype(np.float32))

            new_X_test = np.zeros((test_X.shape[0], test_X.shape[1], n_components))
            for i in range(test_X.shape[1]):
                new_X_test[:, i, :] = pca_scalers[i].transform(test_X[:, i, :]) 
            print('finished test x pca')



    #         return X,y, X_val, y_val, new_X_test
            ## END TRAINING AND TESTING DATA 


            if first_model:
                print('building model')
                smaller_model = build_keras_model(X.shape[1],X.shape[2])
                print(smaller_model.summary())

            # test again at 700 epochs
            if first_model:
                start = time.time()
                # 800 epochs
    #             early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, restore_best_weights=True)
                print('starting training')
                history = smaller_model.fit(x=X, y=y, batch_size=input_batch_size, epochs=200, verbose=input_verbose, 
                          validation_data=(X_val, y_val), shuffle=False,  use_multiprocessing=False, callbacks=[])
                end=time.time()

                print((end-start)/60,' minutes')
            else:

                # Freeze the layers except the last 5 layers
                for layer in smaller_model.layers[:-3]:
                    layer.trainable = False
                # Check the trainable status of the individual layers

    #             for layer in smaller_model.layers:
    #                 print(layer, layer.trainable)

    #             smaller_model.compile(optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='mse', metrics=['mse'])

                start = time.time()
    #             early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, restore_best_weights=True)
                smaller_model = build_keras_model(X.shape[1],X.shape[2])
                print(smaller_model.summary())
                history = smaller_model.fit(x=X, y=y, batch_size=input_batch_size, epochs=200, verbose=input_verbose, 
                          validation_data=(X_val, y_val), shuffle=False,  use_multiprocessing=False, callbacks=[])
                end=time.time()
                print((end-start)/60,' minutes')


            history_df  = pd.DataFrame(history.history)
            history_df[['mse', 'val_mse']].iloc[-100:, :].plot()
            plt.show()
            prediction = smaller_model.predict(new_X_test)[0][0] 

            if idx % print_idx==0:
                print(f"Prediction = {prediction}")



            last_corr_date = train_y.index.max()
            last_corr = train_y[train_y.index.max()]  
            if idx % print_idx==0:
                print(f"Last corr = {last_corr}")

            pred_dates.append(most_recent_date)
            predicted_corr.append(prediction)
            last_corr_for_prediction_day.append(last_corr)

            if input_verbose==1 and idx % print_idx==0:
                print(f"{stock_pairing} corr7-day corr of close from {most_recent_date} to {prediction_end} is {prediction} ")

            first_model = False


        end = time.time()

        print(f"Predictions took {(end-start)/60} mins")

        squarred_difference = (np.array(last_corr_for_prediction_day)-np.array(predicted_corr))**2

        prediction_df = pd.DataFrame({ 'pred_date_start':pred_dates,'stock_pair':pairs_of_stocks,   'first_stock_industry': first_stock_industries, 
                       'second_stock_industry': second_stock_industries,
                       'predicted_corr': predicted_corr, 'last_7_day_corr_for_pred_date_start': last_corr_for_prediction_day, 
                'squarred_diff_7_day_cor': (np.array(last_corr_for_prediction_day)-np.array(predicted_corr))**2
                     })

        if job_id:
            tmp_filepath = '../data/lstm_tmp_prediction_dfs'
            if not os.path.isdir(f"{tmp_filepath}"):
                os.mkdir(f"{tmp_filepath}")
            prediction_df.to_csv(
            f'{tmp_filepath}/{job_id}_lstm_test_predictions_{most_recent_date}-{prediction_end}.csv', index=False)
        else:
            prediction_df.to_csv(
        f'../data/predictions/lstm_test_predictions_{most_recent_date}-{prediction_end}.csv', index=False)

            
            

    prediction_for_upcoming_week(final_stock_df, stock_pairs, job_id=job_id)