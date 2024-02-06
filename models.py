import torch
import torch.nn as nn
import os


MODELS_SAVE_DIR = 'models'
if not os.path.isdir(MODELS_SAVE_DIR):
    os.mkdir(MODELS_SAVE_DIR)


class ValueNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, lr,
                 model_name:str='untitled', load_saved_model=True):
        super().__init__()
        #creation des neurones et du nombres de leurs couches 3 pour les 3 positions possibles
        self.fc1 = nn.Linear(input_size, hidden_size) #dimmensionement du reseau de neuronnes
        self.fc2 = nn.Linear(hidden_size, hidden_size)#nombre de couches
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.loss_fn = nn.MSELoss() #on crée la fonction perte mean squared error loss qui calcule l erreur moyenne
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr) #on etablit notre fonction d optimisation
#self.parameters() est utilisé pour passer tous les paramètres apprenables du modèle à l'optimiseur Adam.



    def forward(self, x): #les entrées traverses couches par couches les neurones afin de donner une sortie , cela permet donc de calculer les prédictions du modèle et la perte résultante
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def train_on_datapoint(self, V_S, V_S_a_max):
        self.optimizer.zero_grad() #il est important de remettre à zéro les gradients des paramètres du réseau, car par défaut, les gradients en PyTorch s'accumulent

        loss = self.loss_fn(V_S, V_S_a_max) #fonction perte qui va calculer l'ecart entre la valeur réel (l'etat actuelle) et celle voulue l etat souhaité ( la valeur maximal d'action )


        loss.backward()
        # cette commande calcule combien chaque paramètre du modèle doit être ajusté pour minimiser l'erreur de prédiction (la perte).
        #les gradients sont calculés en utilisant le calcul différentiel, en remontant depuis la sortie du réseau jusqu'aux couches d'entrée.

        self.optimizer.step() #On a le calcule du gradients de chaques poids et on effectue une mise à jour des poids du modèle


        #y=f(∑ i=1 à n (w.i*x.i)+b) avec w le poids des neuronnes x l entrée et y la sortie

