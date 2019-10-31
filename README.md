# rbf_for_tf2
*Author: Petra Vidnerová, The Czech Academy of Sciences, Institute of Computer Science*


RBF layer for Tensorflow 2.0 (as custom layer derived from tf.keras.layers.Layer)

You need rbflayer.py to use RBF layers in your code. See test.py for
**very simple** examples.

Feel free to use or modify the code. 

## Requirements:
 Tensorflow, Scikit-learn, optionally Matplotlib (only for the toy example in test.py)

## Usage:

```
  # creating RBF network
  rbflayer = RBFLayer(10,
                      initializer=InitCentersRandom(X),
                      betas=2.0,
                      input_shape=(num_inputs,))

  model = Sequential()
  model.add(rbflayer)
  model.add(Dense(n_outputs, use_bias=False))
``` 

or using KMeans clustering for RBF centers 

```
  # creating RBFLayer with centers found by KMeans clustering
  rbflayer = RBFLayer(10,
                      initializer=InitCentersKMeans(X),
                      betas=2.0,
                      input_shape=(num_inputs,))
``` 
 
 If you need any other setup of centers or widhts, you can very easily define your own initializer,
 just write your subclass of `tensorflow.keras.initializers.Initializer`.

 Because you have created Keras model with a custom layer, you need to take it into 
 account if you need to save it to file and load it.
 Saving is no problem:
 ```
 model.save("some_fency_file_name.h5")
 ```
 but while loading you have to specify your custom object RBFLayer:
 ```
 rbfnet = load_model("some_fency_file_name.h5", custom_objects={'RBFLayer': RBFLayer})
 ```

 You can also load weights (centers or widhts) from file (.npy file with an numpy array of the right shape),
 see IntFromFile in initializer.py and
 example in test.py.


## See also:
**[Old repo](https://github.com/PetraVidnerova/rbf_keras/)** that was written
in 2017 for Keras. 


## Contact:
If you need help, do not hesitate to contact me via petra@cs.cas.cz or write an Issue.

## How to cite:
In case you use this RBF layer for any experiments that result in publication, please consider citing it. Thanks :heart:

*Vidnerová, Petra. RBF-Keras: an RBF Layer for Keras Library. 2019. 
Available at https://github.com/PetraVidnerova/rbf_keras*

**Thanks** to the author of the very first citation:   Lukas Brausch, et al. Towards a wearable low-cost ultrasound device for classification of muscle activity and muscle fatigue. 2019 
[doi:10.1145/3341163.3347749](https://doi.org/10.1145/3341163.3347749)



## Acknowledgement: 
This work  was partially supported by the Czech Grant Agency grant 18-23827S 
and institutional support of the Institute of Computer Science RVO 67985807.

