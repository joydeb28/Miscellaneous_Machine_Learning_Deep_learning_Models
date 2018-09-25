import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


dataset = np.loadtxt('data/normalized_apple_prices.csv')



def window_transform_series(series,window_size):
    X = []
    y = []
    
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i])
        y.append(series[i])
        
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y


def create_model():
    
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    
    return model

def plot():
    plt.plot(dataset)
    plt.xlabel('time period')
    plt.ylabel('normalized series value')


def train_model(model):
    model.fit(X_train, y_train, epochs=500, batch_size=64, verbose=1)
    return model

def predict(model,x):
    predict = model.predict(x)
    return predict

def plot_dataset():
    plt.plot(dataset,color = 'k')
    


def prediction_plot():
    plt.plot(np.arange(window_size,split_pt,1),train_predict,color = 'b')
    
    plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')
    
    plt.xlabel('day')
    plt.ylabel('(normalized) price of Apple stock')
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()



window_size = 7
X,y = window_transform_series(series = dataset,window_size = window_size)


train_test_split = int(np.ceil(2*len(y)/float(3))) 

X_train = X[:train_test_split,:]
y_train = y[:train_test_split]

X_test = X[train_test_split:,:]
y_test = y[train_test_split:]
print("X_train:",X_train,",X_train.shape[0]: ",X_train.shape[0])
X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))
'''

np.random.seed(0)

model = create_model()
model = train_model(model)

train_predict = predict(model,X_train)
test_predict = predict(model,X_test)

split_pt = train_test_split + window_size 

plot_dataset()
print()
prediction_plot()


'''





























