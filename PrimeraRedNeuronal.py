import tensorflow as tf
import numpy as np

celsius = np.array([-40, - 10, 0, 8, 15, 22, 38], dtype= float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])

modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.01),
    loss = 'mean_squared_error'
)

print('Comenzando Entrenamiento...')

record = modelo.fit(celsius, fahrenheit, epochs = 7000, verbose = False)

print('Modelo Entrenado!')

print('Hagamos una prediccion')

PruebaDePrediccion = modelo.predict(np.array([15]))

print('El resultado es ', str(PruebaDePrediccion),'Fahrenheit')

import matplotlib.pyplot as plt
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de Perdida')
plt.plot(record.history['loss'])

print('Variables del Modelo')
print(capa.get_weights())
