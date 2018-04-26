# Intro a Redes Neuronales Artificiales con Keras (*Intro to ANNs w/Keras*)

> **Author:** Rodolfo Ferro Pérez <br/>
> **Email:** [ferro@cimat.mx](mailto:ferro@cimat.mx) <br/>
> **Twitter:** [@FerroRodolfo](http://twitter.com/FerroRodolfo) <br/>
> **GitHub:** [RodolfoFerro](https://github.com/RodolfoFerro) <br/>

## About 🕹

[FLISoL](https://flisol.info/FLISOL2018/Mexico/Leon) is the largest Free Software dissemination event in Latin America and is aimed at all types of audiences: students, academics, businessmen, workers, public officials, enthusiasts and even people who do not have much computer knowledge.

This is a Python 🐍 workshop for the *'Festival Latinoamericano de Instalación de Software Libre 2018'* ([FLISoL](https://flisol.info/FLISOL2018/Mexico/Leon)) at the *Instituto Tecnológico de León*, for which I was invited.

It is basically a 101 workshop about [*Artificial Neural Networks*](https://en.wikipedia.org/wiki/Artificial_neural_network) using [Keras](https://keras.io/).


## Setup ⚙️

We'll be working on [Azure Notebooks](https://notebooks.azure.com/) for which you can create and use a free account to train ANN models online.

A new library will be needed, and for this, you can import all the code from this repo: https://github.com/RodolfoFerro/FLISoL18


## Content 👾

All repo content is contained inside the [main](https://github.com/RodolfoFerro/FLISoL18/tree/master/main) folder, in which you'll find a set of Jupyter Notebooks with the ANN code, along with a pre-trained model and a set of images use in the Notebooks.

Demo Gist for [PerceptronFLISoL18.py](https://github.com/RodolfoFerro/FLISoL18/blob/master/PerceptronFLISoL18.py) and [SigmoidFLISoL18.py](https://github.com/RodolfoFerro/FLISoL18/blob/master/SigmoidFLISoL18.py): https://gist.github.com/RodolfoFerro/46dc7ba3dded4cd6a3a9d58e1284557a


#### PerceptronFLISoL18
```python
import numpy as np


class PerceptronFLISoL():
    def __init__(self, entradas, pesos):
        """Constructor de la clase."""
        self.n = len(entradas)
        self.entradas = np.array(entradas)
        self.pesos = np.array(pesos)

    def voy_no_voy(self, umbral):
        """Calcula el output deseado."""
        si_no = (self.entradas @ self.pesos) >= umbral
        if si_no:
            return "Sí voy."
        else:
            return "No voy."


if __name__ == '__main__':
    entradas = [1, 1, 1, 1]
    pesos = [-4, 3, 1, 2]

    dev = PerceptronFLISoL(entradas, pesos)
    print(dev.voy_no_voy(3))
```

#### SigmoidFLISoL18
```python
import numpy as np


class SigmoidNeuron():
    def __init__(self, n):
        np.random.seed(123)
        self.synaptic_weights = 2 * np.random.random((n, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_output, iterations):
        for iteration in range(iterations):
            output = self.predict(training_inputs)
            error = training_output.reshape((len(training_inputs), 1)) - output
            adjustment = np.dot(training_inputs.T, error *
                                self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment

    def predict(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == '__main__':
    # Initialize Sigmoid Neuron:
    sigmoid = SigmoidNeuron(2)
    print("Inicialización de pesos aleatorios:")
    print(sigmoid.synaptic_weights)

    # Datos de entrenamiento:
    training_inputs = np.array([[1, 0], [0, 0], [0, 1]])
    training_output = np.array([1, 0, 1]).T.reshape((3, 1))

    # Entrenamos la neurona (100,000 iteraciones):
    sigmoid.train(training_inputs, training_output, 100000)
    print("Nuevos pesos sinápticos luego del entrenamiento: ")
    print(sigmoid.synaptic_weights)

    # Predecimos para probar la red:
    print("Predicción para [1, 1]: ")
    print(sigmoid.predict(np.array([1, 1])))
```

***

### ABOUT COPYING OR USING PARTIAL INFORMATION: 🔐
* These documents were originally created by the author.
* Any usage of these documents or their contents is granted according to the provided license and its conditions.
* For any question, you can contact the author via email or Twitter.

**Copyright (c) 2018 Rodolfo Ferro**
