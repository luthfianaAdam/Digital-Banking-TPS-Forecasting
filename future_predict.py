import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from dateutil import relativedelta, parser


def future_predict(inputfile, model_filename, predict_range):
    model = tf.keras.saving.load_model(model_filename)
    df = pd.read_csv(inputfile) # Replace with your filename

    # Select only the "tps" column
    data = df[['tps']]
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df.set_index('tanggal', inplace=True)

    # Convert the dataframe to a numpy array
    dataset = data.values

    # print(df)
    print(dataset)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    # Split the data into training set and test set
    print(scaled_data)
    

    # Create sequences of data points
    # Define the lookback period and split into samples
    lookback = 2
    n_input = 10
    real_data_len = len(dataset)


    test_data = scaled_data[real_data_len - n_input - lookback: , :]
    # print(test_data)

    x_test = []
    y_test = dataset[real_data_len:, :]
    for i in range(lookback, len(test_data)):
        x_test.append(test_data[i-lookback:i, 0])

    # print(x_test)
    # print(y_test)
        
    # Plot the data
    train = df[:real_data_len-n_input]
    train = train[['tps']]
    result = train.reset_index()
    result['predict'] = None
    print(result)

    # Convert the data to a numpy array
    x_test = np.array(x_test)
    # print(x_test)
    print(f'The array shape is: {np.shape(x_test)}')

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
    print(x_test)
    print(f'The array shape is: {np.shape(x_test)}')

    for i in range(predict_range):
        predictions = model.predict(x_test)
        last_predict = predictions[0]
        print(predictions)
        predictions = scaler.inverse_transform(predictions)
        next_month = result.iloc[-1,0] + relativedelta.relativedelta(months=1)
        result.loc[len(result.index)] = [pd.to_datetime(next_month), None, predictions[0][0]]

        x_test = list(x_test)
        x_test.append(np.array(x_test[-1]))
        x_test[-1][lookback-1] = last_predict
        x_test.pop(0)
        x_test = np.array(x_test)
        print(x_test)        
        
    print(result)
    return result
    

inputfile = 'transaction-datasets.csv'
modelname = 'future_predict_models/model1.keras'
predict_range = 24   # predict next 5 month
result = future_predict(inputfile,modelname,predict_range)
# print(result.iloc[-1,0])



# # # adding new predicted value
# # x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1]))
# # print(x_test)
# # print(f'The array shape is: {np.shape(x_test)}')
# x_test = list(x_test)
# # print(x_test)
# # print(x_test[-1][1])
# # print(f'The array shape is: {np.shape(x_test)}')
# # x_test.append(np.array([[0.89999999],[0.99999999]]))
# x_test.append(np.array([x_test[-1][1],last_predict]))
# x_test.pop(0)
# # print(x_test)
# # print(f'The array shape is: {type(x_test)}')
# # print(f'The array shape is: {np.shape(x_test)}')
# x_test = np.array(x_test)
# # print(x_test)


# print(f'The array shape is: {np.shape(x_test)}')
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
# print(x_test)
# print(f'The array shape is: {np.shape(x_test)}')

# print(test_data[2-2:2])
# print(np.array([[0.89],[0.99]]))
# print(type(test_data))
# print(type(test_data[2-2:2]))


# # Get the models predicted price values 
# predictions = model.predict(x_test)
# print(predictions)
# predictions = scaler.inverse_transform(predictions)

# # Evaluate model
# # Calculate RMSE
# rmse1 = np.sqrt(np.mean(predictions - y_test)**2)
# max = df['tps'].max()
# min = df['tps'].min()
# rmse2 = rmse1/(max-min)

# # Plot the data
# train = df[:training_data_len]
# valid = df[training_data_len:]
# valid['Predictions'] = predictions

# Visualize the data
# plt.figure(figsize=(16,8))
# plt.title('Model')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('TPS', fontsize=18)
# # plt.plot(train.index, train['tps'])
# # plt.plot(valid.index, valid['tps'])
# plt.plot(result['tanggal'], result[['tps','predict']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()