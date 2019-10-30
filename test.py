import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
import matplotlib.pyplot as plt


def load_data():

    data = np.loadtxt("data/data.txt")
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


def test(initializer=InitCentersRandom):

    X, y = load_data()

    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=initializer(X),
                        betas=2.0,
                        input_shape=(1,))
    model.add(rbflayer)
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

    model.fit(X, y,
              batch_size=50,
              epochs=2000,
              verbose=1)

    y_pred = model.predict(X)

    print(model.layers[0].get_weights())

    # show graph 
    plt.plot(X, y_pred)  # prediction 
    plt.plot(X, y)       # response from data 
    plt.plot([-1, 1], [0, 0], color='black') 
    plt.xlim([-1, 1])

    # plot centers 
    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    plt.scatter(centers, np.zeros(len(centers)), s=20*widths)

    plt.show()
    
    # saving to and loading from file 
    filename = f"rbf_{initializer.__name__}.h5"
    print(f"Save model to file {filename} ... ", end="")
    model.save(filename)
    print("OK")

    print(f"Load model from file {filename} ... ", end="") 
    newmodel = load_model(filename, 
                          custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    # check if the loaded model works same as the original 
    y_pred2 = newmodel.predict(X)
    print("Same responses: ", all(y_pred == y_pred2))

if __name__ == "__main__":

    # test simple RBF Network with random  setup of centers
    test()

    # test simple RBF Network with centers set up by k-means
    test(InitCentersKMeans)
