# Neural DSRT: Neural dialogue systems for humans

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/sbarham/dsrt/LICENSE)

Newborn from the University of Maryland, Neural DSRT (pronounced *dessert*) is a high-level neural dialogue systems API, written in __Python__ and running on top of familiar deep-learning (DL) and machine-learning (ML) libraries like [Keras](https://github.com/keras-team/keras), [TensorFlow](https://github.com/tensorflow/tensorflow), and [scikit-learn](https://github.com/scikit-learn/scikit-learn). It focuses on allowing for the easy construction, training, and testing of __neural dialogue models__. 

Its key purpose is *to enable rapid development and experimentation* in a burgeoning field otherwise bereft of high-level libraries with busy researchers in mind.

<!-- Read the documentation at [Keras.io](https://keras.io) -->


## What is it for?

__Neural DSRT__ is all about building end-to-end dialogue systems using state-of-the-art neural dialogue models. It is a new project (born at the University of Maryland in the waining days and weeks of March, 2018), and it still has a lot of growing to do.

In order to help that growth along, we adopt a few guiding principles liberally from [Keras](https://github.com/keras-team/keras):

- __User-friendliness.__ Ease-of-use should be front and center, the library should expose consistent & simple APIs, and should minimize the amount of work involved in getting common use-cases up and running. The focus should be on enabling rapid, hassle-free experimentation with neural dialog models.

- __Modularity.__ Dialogue experiments, and their constituent components -- dataset wrappers, data preprocessors, neural dialogue models, conversation objects -- should alike be implemented as fully-configurable modules that can be plugged together with as few restrictions as possible. Experiments, and their components, should be richly configurable -- but components should fall back on sensible defaults, so that configuration should never be necessary

- __Extensibility.__ New modules should be simple to add (as new classes and functions), and existing modules and scripts should provide ample and __liberally documented__ examples.


## How do I use it? 90 seconds to Neural DSRT

*This quickstart guide has yet to be written -- but it should be coming soon.*
<!--
The core data structure of Keras is a __model__, a way to organize layers. The simplest type of model is the [`Sequential`](https://keras.io/getting-started/sequential-model-guide) model, a linear stack of layers. For more complex architectures, you should use the [Keras functional API](https://keras.io/getting-started/functional-api-guide), which allows to build arbitrary graphs of layers.

Here is the `Sequential` model:

```python
from keras.models import Sequential

model = Sequential()
```

Stacking layers is as easy as `.add()`:

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

Once your model looks good, configure its learning process with `.compile()`:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

If you need to, you can further configure your optimizer. A core principle of Keras is to make things reasonably simple, while allowing the user to be fully in control when they need to (the ultimate control being the easy extensibility of the source code).
```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

You can now iterate on your training data in batches:

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

Alternatively, you can feed batches to your model manually:

```python
model.train_on_batch(x_batch, y_batch)
```

Evaluate your performance in one line:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

Or generate predictions on new data:

```python
classes = model.predict(x_test, batch_size=128)
```

Building a question answering system, an image classification model, a Neural Turing Machine, or any other model is just as fast. The ideas behind deep learning are simple, so why should their implementation be painful?

For a more in-depth tutorial about Keras, you can check out:

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)

In the [examples folder](https://github.com/keras-team/keras/tree/master/examples) of the repository, you will find more advanced models: question-answering with memory networks, text generation with stacked LSTMs, etc.
-->


## How can I install it?

Before attempting to install Neural DSRT, you'll need to install Keras (which it's built on):

- [Keras installation instructions](https://keras.io/#installation)

In installing Keras, you'll of course need to install a neural-network backend. We recommend TensorFlow:

- [TensorFlow installation instructions](https://www.tensorflow.org/install/).

Once you've done this, you're ready to install DSRT. Currently, the only way to do this is from the GitHub source. Thankfully, this is pretty easy, so long as you have `pip` installed on your machine (did we mention you'll need Python? you'll need Python -- we recommend the latest version of Python 3). 

Just in case, instructions for installing `pip` may be found here:

- [pip installation instructions](https://pip.pypa.io/en/stable/installing/)

Assuming you've followed up until this point, proceed to __clone DSRT__ using `git`:

```sh
git clone https://github.com/sbarham/dsrt.git
```

 Now, `cd` to the Neural DSRT folder and install using `pip`:
```sh
cd dsrt
sudo pip install .
```

It's as easy as that. Now you're ready to use Neural DSRT!


# How can I help?

As we mentioned above, DSRT is very young -- in fact, it's only a few weeks old at the moment. If you're a developer (and especially if you're confident with deep learning, machine learning, or neural dialogue systems) and you'd like to help, please contact the original authors directly at `sbarham@cs.umd.edu`. We'd love to collaborate with you.
