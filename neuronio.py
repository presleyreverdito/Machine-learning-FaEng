#'''
#Universidade Federal de Mato Grosso 2019/1
#Aprendizado de Máquina
#Código da primeira aula prática
#Regressão linear com Keras

#autor: raoni teixeira
#'''
#!/usr/bin/python
import keras
from keras.models import Sequential
from keras.layers import Dense,  Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import regularizers
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Base de dados utilizada nos experimentos
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)


#Transformando os dados em Array
Array_train = np.array(X_train)
Array_test = np.array(X_test)

#Calculando a media dos arrays
Media_train = np.average(Array_train)
Max_train = np.max(Array_train)
Min_train = np.min(Array_train)

#Subtraindo a media dos dados
Array_train = (Array_train - Min_train) / (Max_train - Min_train)
Array_test  = (Array_test - Min_train) / (Max_train - Min_train)
#Voltando para DataFrame
Array_train = pd.DataFrame(Array_train)
Array_test = pd.DataFrame(Array_test)





#Fazendo a potencia e a somatoria
#Norm_train = np.vdot(Norm_train,Norm_train)
#Array_train = Array_train/Max_train
#Array_test  = Array_test/Max_train

#print("*************RESULTADO************")
#print(Array_train)
#print("**********************************")


# Visualizando os dados
features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='x')
    plt.title(col)
    #dados = boston.Series(boston.Categorical(['B']))
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()
#print("transformando em array\n")

#boston.to_numpy()
#print(dados)
#print("\n *****************\n")




# Modelos
epochs = 1000
model = Sequential()

# modelo 1 - um neurônio
#model.add(Dense(1, input_dim=2))

#modelo 2 - rede com três camadas

model.add(Dense(5,  input_dim=2, activation = 'relu'))
model.add(Dense(3,  input_dim=2, activation = 'relu'))
#model.add(Dense(2,  activation = 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()




model.fit(x = Array_train, y = Y_train,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)

Y_pred_test = model.predict(Array_test)

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = X_test[col]

    y = Y_test
    plt.scatter(x, y, marker='x')
    plt.title(col)
    #dados = boston.Series(boston.Categorical(['B']))
    plt.xlabel(col)
    plt.ylabel('MEDV')
#plt.show()
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = X_test[col]
    y = Y_pred_test
    plt.scatter(x, y, marker='x')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()
Result = [Y_test, X_test]
print(pd.concat(Result, axis=1))
print(mean_squared_error(Y_test, Y_pred_test))

while(1):
    #   Recebe dois valores de entrada
    X1 = input('Insira LSTAT: ')
    X2 = input('Insira RM: ')

    #   Adiciona os valores normalizados com os dados do treino a uma lista
    S1 = []
    S1.append([(float(X1) - Min_train)/(Max_train - Min_train), (float(X2) - Min_train)/(Max_train - Min_train)])

    #   Converte para DataFrame e envia para a predição
    entrada = pd.DataFrame(S1, columns = ['LSTAT','RM'])
    print (entrada)

    saida = float(model.predict(entrada))
    print(saida)

    Restart = input('Continuar [Y/N]: ')
    if (Restart == 'N'):
        break

plt.plot(Y_test, Y_pred_test, 'bo')
plt.show()
print(Y_test)
print(mean_squared_error(Y_test, Y_pred_test))
