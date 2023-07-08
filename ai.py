import numpy as np

x_enter = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]), dtype=float) # données d'entrer
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # données de sortie /  1 = rouge /  0 = bleu

x_enter = x_enter/np.amax(x_enter, axis=0)

X = np.split(x_enter, [8])[0]
xPrediction = np.split(x_enter, [8])[1]

class NeuralNetwork(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # Matrice 3x2
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # Matrice 3x1

    def forward(self, X):
        
        self.z = np.dot(X, self.W1) 
        self.z2 = self.sigmoid(self.z) 
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    
    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    def backward(self, X, y,o):

        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

    def predict(self):
        print("Donnée prédite apres entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        if(self.forward(xPrediction) < 0.5):
           print("La fleur est BLEU ! \n")
        else:
           print("La fleur est ROUGE ! \n")
        
  


NN = NeuralNetwork()

for i in range(1000): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()


