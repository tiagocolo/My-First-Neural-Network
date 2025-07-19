import tensorflow as tf
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt

def redondear(n):
    return int(Decimal(n).quantize(0, rounding=ROUND_HALF_UP))

celsius = np.array([-40, - 10, 0, 8, 15, 22, 38], dtype= float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])

modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print('Comenzando Entrenamiento...')

record = modelo.fit(celsius, fahrenheit, epochs = 1000, verbose = False)

print('Modelo Entrenado!')

print('Hagamos una prediccion')

PruebaDePrediccion = modelo.predict(np.array([20]))

print('El resultado es ', redondear(float(PruebaDePrediccion)),'Fahrenheit')

plt.xlabel('# Epoca')

plt.ylabel('Magnitud de Perdida')

plt.plot(record.history['loss'])
