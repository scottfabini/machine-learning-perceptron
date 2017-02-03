## Synopsis

Machine Learning using a multi-layer neural network with 784 inputs (corresponding to 28x28
image from MNIST dataset), a configurable number of classifiers in hidden
layer, and 10 output classifiers (corresponding to digits 0-9).

Confusion Matrix showing correlation between target and actual results. High numbers on diagonal indicate correct classification.<br>
![alt tag](http://web.cecs.pdx.edu/~sfabini/img/confusion_matrix.png)

# Requirements

Python 3 installed.

## Installation

```
git clone git@github.com:scottfabini/machine-learning-perceptron.git
cd machine-learning-perceptron/2_multi-layer
```
Download mnist training and test data to this directory from:
http://web.cecs.pdx.edu/~mm/MachineLearningWinter2017/mnist_train.csv
http://web.cecs.pdx.edu/~mm/MachineLearningWinter2017/mnist_test.csv
```
python3 -m pip install numpy
python3 -m pip install scipy
python3 -m pip install scikit-learn
python3 Main.py 
```

## Operation

```
python3 Main.py [-n <hidden_layer_size=20>] [-o <output_layer_size=10>] [-m <momentum=0.9>] [-l
<learning rate=0.1>] [-e <epochs=50>]
```

## License

Copyright (c) 2016 Scott Fabini (scott.fabini@gmail.com)


Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

