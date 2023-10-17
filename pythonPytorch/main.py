#https://www.alura.com.br/artigos/primeiros-passos-com-pytorch
#https://www.geeksforgeeks.org/how-to-visualize-a-neural-network-in-python-using-graphviz/
#https://www.thetechplatform.com/post/visualize-neural-network-in-python
#https://www.devmedia.com.br/como-criar-minha-primeira-classe-em-python/38912
#https://machinelearningmastery.com/activation-functions-in-pytorch/
#https://www.projectpro.io/recipes/optimize-function-adam-pytorch
#https://www.ibm.com/topics/gradient-descent
#https://sites.icmc.usp.br/andre/research/neural/
#https://pytorch.org/docs/stable/tensors.html
#https://www.binarystudy.com/2022/10/pytorch-how-to-cast-tensor-to-another-type-in-PyTorch.html?m=1#:~:text=In%20PyTorch%2C%20we%20can%20cast,dtype%20passed%20as%20the%20parameter.
#https://machinelearningmastery.com/building-a-multiclass-classification-model-in-pytorch/
#https://stackoverflow.com/questions/53723928/attributeerror-series-object-has-no-attribute-reshape

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from sklearn.preprocessing import OneHotEncoder

seeds = pd.read_csv('/Users/mirella/Desktop/mestrado/Redes Neurais Não Supervisionadas/trab/seeds/seeds.txt', sep=r'\t',engine='python')
#print(seeds) #Display the whole dataset
#print(seeds.head())
XX = seeds.iloc[:, 0:8].values #iloc function enables us to select a particular cell of the dataset, that is, it helps us select a value that belongs to a particular row or column from a set of values of a data frame or dataset.
dXX = pd.DataFrame(XX, columns = ['Area', 'Perimeter','Compactness','Kernel Length', 'Kernel Width', 'Asymmetry Coefficient','Kernel Groove Length', 'Original Labels'])
dXX.loc[-1] = [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 1]  # adding a row
dXX.index = dXX.index + 1  # shifting index
dXX.sort_index(inplace=True)
X = seeds.iloc[:, 0:7].values
dX = pd.DataFrame(X, columns = ['Area', 'Perimeter','Compactness','Kernel Length', 'Kernel Width', 'Asymmetry Coefficient','Kernel Groove Length'])
dX.loc[-1] = [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22]  # adding a row
dX.index = dX.index + 1  # shifting index
dX.sort_index(inplace=True)
print("Original Dataframe:\n", dXX)
#print("Dataframe without the Labels:\n", dX)

#print("Size:\n",dXX.size)

#check if the dataset has missing elements
#print(dXX.isnull().sum())

x = dX
y_original = dXX['Original Labels']

#mapping = {1:'Kama', 2:'Rosa', 3:'Canadian'}
#y_original = [mapping[i] for i in dXX['Original Labels']]

#Used to turn the clusters from 1,2,3 to [1,0,0],[0,1,0],[0,0,1]
#y_original.values.reshape(1, -1)
#print("y_original:\n", y_original)

#y_original = pd.DataFrame(y_original)

#yout = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y_original)
#print("yout.categories:\n", yout.categories_)

#y_original = yout.transform(y_original)
#print("y_original:\n", y_original)

x = x.values #transformar o dataframe na matriz correspondente
y_original = y_original.values

#70% training and 30% test - técnica de hold-out
#Now that our data set has been split into training data and test data, we’re ready to start training our model
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y_original, test_size = 0.3, random_state=1)
#x_training_data: variável associada aos dados de treino, y_training_data: variável associada às classes de treino
#x_test_data: variável associada aos dados de teste, y_test_data: variável associada às classes de teste

#Turn the arrays into tensors
x_training_data = torch.FloatTensor(x_training_data)
print("x_training_data", x_training_data)
print("len_x_training_data", np.shape(x_training_data))
x_test_data = torch.FloatTensor(x_test_data)
print("x_test_data", x_test_data)
print("len_x_test_data", np.shape(x_test_data))
y_training_data = torch.FloatTensor(y_training_data)
print("y_training_data", y_training_data)
print("len_y_training_data", np.shape(y_training_data))
y_test_data = torch.FloatTensor(y_test_data)
print("y_test_data", y_test_data)
print("len_y_test_data", np.shape(y_test_data))

#This neural network holds seven inputs and three outputs (Kama, Rosa, Canadian). These are respectively the columns of x and the number of possible classes I have in y.
#Furthermore, two hidden layers were added, one with 10 neurons and the other with 20 neurons. The number of neurons in the hidden layers was chosen arbitrarily.
#To create the network structure, I defined a class that I named Model. Inside the model class I define the init that will start the network structure.
class Model(nn.Module):
    def __init__(self,input=7,hidden_layer1=10,hidden_layer2=20,output=3):
        super().__init__()
        self.fc1 = nn.Linear(input,hidden_layer1) #Applies a linear transformation to the incoming data
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.out = nn.Linear(hidden_layer2, output)

    #Still within the model class, I define the forward function that will be responsible for propagating the network.
    #Propagation is what takes the input to the output. Each of the connections in a network is connected using weights and the output of each layer of the network is made using an activation function.
    #I will use the Rectified Linear Unit (ReLU) activation function which always returns positive values. ReLU is a non-saturating function, which means that it does not become flat at the extremes of the input range.
    #Instead, ReLU simply outputs the input value if it is positive, or 0 if it is negative.
    #This simple, piecewise linear function has several advantages over sigmoid and tanh activation functions.
    #First, it is computationally more efficient, making it well-suited for large-scale neural networks. Second, ReLU has been shown to be less susceptible to the vanishing gradient problem, as it does not have a flattened slope.
    #Plus, ReLU can help sparsify the activation of neurons in a network, which can lead to better generalization.

    #Another activation function that will be employed is Tanh. The outputs values are between -1 and 1, with mean output of 0.
    #This can help ensure that the output of a neural network layer remains centered around 0, making it useful for normalization purposes. Tanh is a smooth and continuous activation function, which makes it easier to optimize during the process of gradient descent.

    #The last activation function that will be utilized is Sigmoid. It takes any input and maps it to a value between 0 and 1, which can be interpreted as a probability.
    #This makes it particularly useful for binary classification tasks, where the network needs to predict the probability of an input belonging to one of two classes.

    #Activation functions are applied to the output of each neuron in a neural network to introduce non-linearity into the model.
    #Without activation functions, neural networks would simply be a series of linear transformations, which would limit their ability to learn complex patterns and relationships in data.
    #Activation functions train a neural network.
    def forward1(self, x):
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.out(x) #ReLU is a powerful activation function for hidden layers but is not suitable for the output layer. This is because it only produces positive values and is not a smooth function, which can lead to unstable predictions.
        # When choosing an activation function for the output layer, it is important to consider the range of the predicted values and the desired properties of the function.
        # Some commonly used activation functions for the output layer include the sigmoid, softmax, and linear functions.

        return x

    def forward2(self, x):
        x = nn.Tanh(x)
        x = nn.Tanh(x)
        x = nn.Tanh(x)

        return x

    def forward3(self, x):
        x = func.sigmoid(x)
        x = func.sigmoid(x)
        x = func.sigmoid(x)

        return x

#Instantiate the network created using the command:
classification_model = Model()

#Check whether the network is taking the input to a result close to the desired output. We do this through an objective function or cost function
objective_function = nn.CrossEntropyLoss()

rate_learning = 0.01
#On the first attempt, the neural network will not obtain a satisfactory output. This happens because the weights that connect each of the neurons are defined randomly.
#Therefore, it is needeed to correct these weights. The optimizer that will be used in this process is defined by:
optimizer = torch.optim.Adam(classification_model.parameters(), lr=rate_learning) #The Adam optimizer is also an optimization techniques used for machine learning and deep learning, and comes under gradient decent algorithm.

#Let's train the neural network. The propagation and backpropagation process will be repeated for 100 cicles.
#Thus, we hope to correct the weights to obtain a network that correctly transforms the input into a prediction of the seed class.
cicles = 110
costs = []
for i in range(cicles):
  y_predicted_data = classification_model.forward1(x_training_data)
  #y_predicted_data = y_predicted_data.flatten() #convert a matrix to an array
  print('y_predicted_data:', y_predicted_data)
  print("len_y_predicted_data", np.shape(y_predicted_data))
  y_training_data = y_training_data.type('torch.LongTensor')
  cost = objective_function(y_predicted_data, y_training_data)
  print("cost:", cost)
  costs.append(cost)

  optimizer.zero_grad() #Here before the backward pass we must zero all the gradients for the variables it will update which are nothing but the learnable weights of the model.
  cost.backward() #Here we are computing the gradients of the loss w.r.t the model parameters.
  optimizer.step() #Here we are calling the step function on an optimizer which will makes an update to its parameters.

#Finally, we can try to predict y values by passing the x_training_data as input. This way we can compare Y and YHat, which is the estimated value.
#The last column of the result table returns 1 for correct estimates and zero for incorrect ones.
preds = []
with torch.no_grad(): #Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward(). It will reduce memory consumption for computations that would otherwise have requires_grad=True.
    for val in x_training_data:
        y_predicted_data = classification_model.forward1(x_training_data)
        #y_predicted_data = classification_model.forward2(x_training_data)
        #y_predicted_data = classification_model.forward3(x_training_data)
        preds.append(y_predicted_data.argmax().item())

print("preds:\n", preds)
df = pd.DataFrame({'Y': y_test_data, 'YHat': preds})
df['Right Predction'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
df