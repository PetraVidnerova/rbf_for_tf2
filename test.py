import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE

import matplotlib.pyplot as plt

from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from initializer import InitFromFile


def load_data():

    data = np.loadtxt("data/data.txt")
    X = data[:, :-1]  # except last column
    y = data[:, -1]  # last column only
    return X, y


def test(X, y, initializer):

    title = f" test {type(initializer).__name__} "
    print("-"*20 + title + "-"*20)

    # create RBF network as keras sequential model
    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=initializer,
                        betas=2.0,
                        input_shape=(1,))
    outputlayer = Dense(1, use_bias=False)

    model.add(rbflayer)
    model.add(outputlayer)

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

    # fit and predict
    model.fit(X, y,
              batch_size=50,
              epochs=2000,
              verbose=0)

    y_pred = model.predict(X)

    # show graph
    plt.plot(X, y_pred)  # prediction
    plt.plot(X, y)       # response from data
    plt.plot([-1, 1], [0, 0], color='black')  # zero line
    plt.xlim([-1, 1])

    # plot centers
    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    plt.scatter(centers, np.zeros(len(centers)), s=20*widths)

    plt.show()

    # calculate and print MSE
    y_pred = y_pred.squeeze()
    print(f"MSE: {MSE(y, y_pred):.4f}")

    # saving to and loading from file
    filename = f"rbf_{type(initializer).__name__}.h5"
    print(f"Save model to file {filename} ... ", end="")
    model.save(filename)
    print("OK")

    print(f"Load model from file {filename} ... ", end="")
    newmodel = load_model(filename,
                          custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    # check if the loaded model works same as the original
    y_pred2 = newmodel.predict(X).squeeze()
    print("Same responses: ", all(y_pred == y_pred2))
    # I know that I compared floats, but results should be identical

    # save, widths & weights separately
    np.save("centers", centers)
    np.save("widths", widths)
    np.save("weights", outputlayer.get_weights()[0])


def test_init_from_file(X, y):

    print("-"*20 + " test init from file " + "-"*20)

    # load the last model from file
    filename = f"rbf_InitFromFile.h5"
    print(f"Load model from file {filename} ... ", end="")
    model = load_model(filename,
                       custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    res = model.predict(X).squeeze()  # y was (50, ), res (50, 1); why?
    print(f"MSE: {MSE(y, res):.4f}")

    # load the weights of the same model separately
    rbflayer = RBFLayer(10,
                        initializer=InitFromFile("centers.npy"),
                        betas=InitFromFile("widths.npy"),
                        input_shape=(1,))
    print("rbf layer created")
    outputlayer = Dense(1,
                        kernel_initializer=InitFromFile("weights.npy"),
                        use_bias=False)
    print("output layer created")

    model2 = Sequential()
    model2.add(rbflayer)
    model2.add(outputlayer)

    res2 = model2.predict(X).squeeze()
    print(f"MSE: {MSE(y, res2):.4f}")
    print("Same responses: ", all(res == res2))


if __name__ == "__main__":

    X, y = load_data()

    # test simple RBF Network with random  setup of centers
    test(X, y, InitCentersRandom(X))

    # test simple RBF Network with centers set up by k-means
    test(X, y, InitCentersKMeans(X))

    # test simple RBF Networks with centers loaded from previous
    # computation
    test(X, y, InitFromFile("centers.npy"))

    # test InitFromFile initializer
    test_init_from_file(X, y)
