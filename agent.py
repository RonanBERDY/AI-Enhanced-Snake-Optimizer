import snake
import models
import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class Agent:
    #valeurs empyriques
    MODEL_HIDDEN_SIZE = 64  #valeur definie par pytorch en fonction des entrées
    LR = 0.001 #taux d apprentissage doit etre petit car c est le step pour garantir la convergence si il est trop petit on va mettre enormement de temps et trop grand on risque de le dépasser
    GAMMA = 0.9 #facteur de décompte
    #proche de 1 , vers 0 seul la récompense immédiate compte et inferieur a 1 recompense immediate plus importante que le futur mais on tient compte des récompenses a venir
    EPOCHS = 100


#systeme de récompense
    APPLE_EAT_REWARD = 10
    APPLE_NO_EAT_REWARD = -10
    SNAKE_DEATH_REWARD = -100
    DISTANCE_REWARD_MULTIPLIER = 1
    SNAKE_LENGTH_REWARD_MULTIPLIER = 0.1
    DEATH_PENALTY_REDUCTION = 0.5
    DEATH_PENALTY_REDUCTION_LIMIT = 10
    record=0
    SNAKE_START_SIZE = 1

    def __init__(self):
        #initialisation du modèle
        self.model_hidden_size = self.MODEL_HIDDEN_SIZE
        self.lr = self.LR
        self.gamma = self.GAMMA
        self.death_count = 0
        self.init_game()
        self.record=self.record
        # Initialize the model and load the saved state
        self.value_nn = models.ValueNN(6, self.model_hidden_size, 1, self.lr,
                                       model_name='snake_valueNN', load_saved_model=True)

    def init_game(self):
        self.game_metrics = {'score': [], 'turns_played': [], 'game_': []}
        #initialisation du jeu
        self.game = snake.Game()
        self.game.snake_size = self.SNAKE_START_SIZE
        self.game.to_call_on_game_quit.append(self.on_quit)



    def get_state(self, snake_head_pos, snake_body_pos_list, apple_pos, move_direction):
        game = self.game
 #on calcule l'etat actuel et la position de la pomme par rapport au serpent
        snakes_forward_direction = game.DIRECTIONS_VALS[move_direction]
        snakes_left_direction = game.DIRECTIONS_VALS[(move_direction-1)%4]
        snakes_right_direction = game.DIRECTIONS_VALS[(move_direction+1)%4]
#nombre d'action possible ici c est les position possible on supprime faire un pas en arrière sinon le serpent meurt directement
        apple_pos_list = [apple_pos,
                          apple_pos-2*np.array((apple_pos[0], 0)),
                          apple_pos-2*np.array((0, apple_pos[1])),
                          apple_pos+2*np.array((self.game._GRID_SIZE-apple_pos[0], 0)),
                          apple_pos+2*np.array((0, self.game._GRID_SIZE-apple_pos[1])),
                          ]

        apple_relative_head = apple_pos_list - snake_head_pos
#calcule de la proximité du corps de la pommes dans toutes les directions possibles
        food_front = min(np.dot(apple_relative_head, snakes_forward_direction))/game._GRID_SIZE
        food_left = min(np.dot(apple_relative_head, snakes_left_direction))/game._GRID_SIZE
        food_right = min(np.dot(apple_relative_head, snakes_right_direction))/game._GRID_SIZE

#calcule de la proximité du corps du serpent dans toutes les directions possibles
        if snake_body_pos_list:
            body_front = min(np.dot(snake_body_pos_list, snakes_forward_direction))/game._GRID_SIZE
            body_left = min(np.dot(snake_body_pos_list, snakes_left_direction))/game._GRID_SIZE
            body_right = min(np.dot(snake_body_pos_list, snakes_right_direction))/game._GRID_SIZE
        else:
            body_front = body_left = body_right = 1
#on envoit ces entrées au modèles
        return torch.Tensor((food_front, food_left, food_right, body_front, body_left, body_right))

    def train(self, game_count):#fonction qui fait  la loop et qui fait l increntation
        """One epoch is one snake life"""
        self.current_game = 0
        game_steps = 0

        while self.current_game < game_count:

            game_steps += 1

            if not self.train_step():


                self.game.snake_size = self.SNAKE_START_SIZE

                self.current_game += 1
                self.game_metrics['turns_played'].append(game_steps)
                game_steps = 0

    def plot_scores(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Calculer la moyenne, le minimum et le maximum des scores
        mean_score = np.mean(self.game_metrics['score'])
        min_score = np.min(self.game_metrics['score'])
        max_score = np.max(self.game_metrics['score'])

        # Tracer le score
        color = 'tab:blue'
        ax1.set_xlabel('Nombre de parties')
        ax1.set_ylabel('Score', color=color)
        ax1.plot(self.game_metrics['score'], label='Score par partie', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=mean_score, color=color, linestyle='--', label='Score moyen')
        ax1.axhline(y=min_score, color='tab:red', linestyle=':', label='Score minimum')
        ax1.axhline(y=max_score, color='tab:purple', linestyle=':', label='Score maximum')

        # Tracer les étapes (tours) par partie
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position(('outward', 60))  # Décaler la colonne de droite de ax2
        color = 'tab:green'
        ax2.set_ylabel('Pas du serpent par partie', color=color)
        ax2.plot(self.game_metrics['turns_played'], label='Étapes par partie', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # Pour éviter que le label de droite soit légèrement coupé
        plt.title('Scores et Étapes du Jeu de Serpent au Fil du Temps avec Valeurs Moyennes, Minimales et Maximales')
        plt.legend()
        plt.show()



    def calculate_distance(self, pos1, pos2):
        return np.sum(np.abs(pos1 - pos2))  # Manhattan distance

    def get_distance_reward(self, new_head_pos, apple_pos, previous_distance):
        new_distance = self.calculate_distance(new_head_pos, apple_pos)
        return self.DISTANCE_REWARD_MULTIPLIER * (previous_distance - new_distance)

    def get_dynamic_apple_reward(self, snake_length):
        return self.APPLE_EAT_REWARD + self.SNAKE_LENGTH_REWARD_MULTIPLIER * (snake_length - 1)

    def get_dynamic_death_penalty(self):
        return self.SNAKE_DEATH_REWARD + self.DEATH_PENALTY_REDUCTION * max(0, self.DEATH_PENALTY_REDUCTION_LIMIT - self.death_count)


    def assess_risk(self, new_head_pos, snake_body_pos_list):
        # Simplified risk assessment: count how close the new head position is to the body
        risk_penalty = 0
        for body_pos in snake_body_pos_list:
            if np.linalg.norm(new_head_pos - body_pos, ord=1) <= 2:  # Using Manhattan distance
                risk_penalty += 10  # Increase penalty for each close body part
        return -risk_penalty  # Return as negative value for penalty

    def assess_escape_routes(self, new_head_pos):
        # Simplified escape route assessment: count the number of free adjacent positions
        escape_route_reward = 0
        for direction in [self.game.DIRECTIONS_VALS[i] for i in range(4)]:  # Assuming 4 directions: up, right, down, left
            adjacent_pos = (new_head_pos + direction) % self.game._GRID_SIZE  # Wrap around grid edges
            if not any(np.array_equal(adjacent_pos, pos) for pos in self.game.snake_positions):
                escape_route_reward += 1  # Reward for each open adjacent position
        return escape_route_reward




    def train_step(self):
        game = self.game
        game.step_game_loop(allow_events=False)

    #on calcule tout les états possibles suivants et les positions possitions possible de la pomme par rapport au serpent
        snake_head_pos = game.snake_positions[0]

        # List of positions occupied by the snake's body excluding the head
        snake_body_pos_list = game.snake_positions[1:]


        # Position of the apple
        apple_pos = game.apple_pos
        previous_distance = self.calculate_distance(snake_head_pos, apple_pos)
        # Current movement direction of the snake
        move_direction = game.direction

        # Now use these variables to get the current state
        current_state = self.get_state(snake_head_pos, snake_body_pos_list, apple_pos, move_direction)

        snakes_forward_direction = game.DIRECTIONS_VALS[move_direction]
        snakes_left_direction = game.DIRECTIONS_VALS[(move_direction-1)%4]
        snakes_right_direction = game.DIRECTIONS_VALS[(move_direction+1)%4]

        snake_head_pos_on_move_forward = (snake_head_pos+snakes_forward_direction) % game._GRID_SIZE
        snake_head_pos_on_move_left = (snake_head_pos+snakes_left_direction) % game._GRID_SIZE
        snake_head_pos_on_move_right = (snake_head_pos+snakes_right_direction) % game._GRID_SIZE

        current_state = self.get_state(snake_head_pos, snake_body_pos_list, apple_pos, move_direction)

        state_after_actions = [
            self.get_state(snake_head_pos_on_move_forward, snake_body_pos_list, apple_pos, move_direction),
            self.get_state(snake_head_pos+snakes_left_direction, snake_body_pos_list, apple_pos, (move_direction-1)%4),
            self.get_state(snake_head_pos+snakes_right_direction, snake_body_pos_list, apple_pos, (move_direction+1)%4)
        ]
 #ici on a calculé les 3(car 3 actions possibles) possibilités d'etat après une action
        V_S = self.value_nn(current_state)#on donne l'etat actuel au modele
        V_S_a = [self.value_nn(state) for state in state_after_actions] #on envoit l'etat après les 3 actions possibles aux neurones
#on met a jour les récompenses:

#ici lorsque le serpent s approche trop de lui

        risk_penalties = [
            self.assess_risk(pos, self.game.snake_positions[1:])
            for pos in [snake_head_pos_on_move_forward, snake_head_pos_on_move_left, snake_head_pos_on_move_right]
        ]
        escape_route_rewards = [
            self.assess_escape_routes(pos)
            for pos in [snake_head_pos_on_move_forward, snake_head_pos_on_move_left, snake_head_pos_on_move_right]
        ]




        R_S_a_1 = [
            self.get_dynamic_apple_reward(game.snake_size) if (apple_pos == pos).all() else self.APPLE_NO_EAT_REWARD
            for pos in [snake_head_pos_on_move_forward, snake_head_pos_on_move_left, snake_head_pos_on_move_right]
        ]

        R_S_a_2 = [
            self.get_dynamic_death_penalty() if game.intersects_body(pos) else 0
            for pos in [snake_head_pos_on_move_forward, snake_head_pos_on_move_left, snake_head_pos_on_move_right]
        ]

        distance_rewards = [
            self.get_distance_reward(pos, apple_pos, previous_distance)
            for pos in [snake_head_pos_on_move_forward, snake_head_pos_on_move_left, snake_head_pos_on_move_right]
        ]

        V_S_a_with_R_S = [R_S_a_1[i] + R_S_a_2[i] + distance_rewards[i]+risk_penalties[i] + escape_route_rewards[i] + self.gamma * V_S_a[i] for i in range(3)]#On calcule la valeur d'action ( équation de Bellman) qui est décomposé en 2 parties ,
        #récompense immédiate et valeur décompté de l'état suivant
        self.value_nn.train_on_datapoint(V_S, max(V_S_a_with_R_S)) #on met a jour les neurones via la fonction train_on_datapoint

        action_probabilities = torch.softmax(torch.Tensor(V_S_a_with_R_S), 0).numpy()#on convertie les vecteurs de nombres à des vecteurs de probabilités

        chosen_action = np.random.choice((0, 1, 2), p=action_probabilities)#on choisit l'action et chacune de ses actions est défini par une probabilité définie plus haut
#ici on fait le choix d'une approche stochastique plutôt que deterministe puisque la localisation de la pomme est aleatoire

        if chosen_action == 1: game.turn_left()
        elif chosen_action == 2: game.turn_right()

        game_score = game.snake_size
        if not game.move():

            self.game_metrics['score'].append(game_score)
            print(f"Game ended. Score: {game_score}")
            if game_score > self.record:
                self.record = game_score
                self.value_nn.save()  # Save the model du score le plus élevé

            return False

        return True

    def on_quit(self):

        quit()

if __name__ == '__main__':
    agent = Agent()
    agent.init_game()
    agent.train(500)
    agent.plot_scores()
