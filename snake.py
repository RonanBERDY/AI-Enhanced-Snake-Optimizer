import pygame
import numpy as np

# Define colors
# Define colors
BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
RED = np.array((255, 0, 0))
GREEN = np.array((0, 255, 0))
BLUE = np.array((0, 0, 255))
# Initialize Pygame
pygame.init()



# Classe pour afficher les FPS (images par seconde) dans la fenêtre de jeu
class FPSDisplay:
    def __init__(self, game):
        self.game = game  # Référence à l'objet de jeu
        self.font = pygame.font.SysFont(None, 25)  # Création d'une police pour le texte
        self.target_fps = 1000  # Valeur cible des FPS

    def draw(self):
        # Calculer et afficher les FPS actuels
        fps = int(self.game.clock.get_fps())  # Obtenir les FPS actuels
        fps_text = self.font.render(f"FPS: {fps} / {self.target_fps}", True, WHITE)  # Rendre le texte des FPS
        self.game.window.blit(fps_text, (10, 10))  # Afficher le texte dans la fenêtre de jeu

# Classe principale pour le jeu de serpent (Snake)
class Game:
    CELL_SIZE = 30  # Taille de chaque cellule dans la grille
    _GRID_SIZE = 20  # Taille interne de la grille

    GRID_SIZE = np.array((_GRID_SIZE, _GRID_SIZE))  # Calcul de la taille de la grille
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE  # Calcul de la taille de la fenêtre

    DIRECTIONS = ['LEFT', 'UP', 'RIGHT', 'DOWN']  # Directions possibles pour le serpent
    DIRECTIONS_VALS = [np.array(i) for i in ([-1, 0], [0, -1], [1, 0], [0, 1])]  # Valeurs associées aux directions

    APPLE_COLOR = RED  # Couleur de la pomme
    SNAKE_COLOR = GREEN  # Couleur du serpent

    def __init__(self):
        self.window = pygame.display.set_mode(self.WINDOW_SIZE)  # Initialisation de la fenêtre de jeu
        pygame.display.set_caption("Snake Game")  # Titre de la fenêtre

        # Chargement et dimensionnement de l'image de fond
        self.background_image = pygame.image.load('snake.jpg')  # Charger l'image
        self.background_image = pygame.transform.scale(self.background_image, self.WINDOW_SIZE)  # Ajuster la taille

        self.clock = pygame.time.Clock()  # Horloge pour contrôler les FPS

        self.fps = 1000  # FPS cible pour le jeu
        self.fps_display = FPSDisplay(self)  # Affichage des FPS

        self.running = True  # État de fonctionnement du jeu

        self.init_new_game()  # Initialisation d'une nouvelle partie

        self.to_call_on_draw = [self.fps_display.draw]  # Fonctions à appeler pour le rendu
        self.to_call_on_game_quit = []  # Fonctions à appeler lors de la fermeture du jeu

    def init_new_game(self):
        # Réinitialise le jeu à son état initial
        self.snake_size = 1  # Taille initiale du serpent
        self.snake_positions = [self.get_random_pos()]  # Position initiale du serpent
        self.direction = 0  # Direction initiale

        self.new_apple()  # Placer une nouvelle pomme

    def get_random_pos(self):
        # Générer une position aléatoire sur la grille
        return np.random.randint(0, self._GRID_SIZE, size=(2,))

    def new_apple(self):
        # Placer une nouvelle pomme à une position aléatoire qui ne croise pas le corps du serpent
        self.apple_pos = self.get_random_pos()
        while self.intersects_body(self.apple_pos):
            self.apple_pos = self.get_random_pos()

    def turn_right(self):
        # Tourner le serpent à droite
        self.direction = (self.direction+1) % 4

    def turn_left(self):
        # Tourner le serpent à gauche
        self.direction = (self.direction-1) % 4

    @property
    def current_move_direction(self):
        # Obtenir la direction actuelle du mouvement du serpent
        return self.DIRECTIONS_VALS[self.direction]

    def intersects_body(self, position):
        # Vérifier si une position donnée intersecte le corps du serpent
        return any(np.array_equal(position, pos) for pos in self.snake_positions)

    def move(self):
        # Déplacer le serpent et gérer les règles du jeu
        next_pos = (self.snake_positions[0] + self.current_move_direction) % self.GRID_SIZE #le serpent se déplace uniquement
        #dans l'interface du jeu
        if self.intersects_body(next_pos): #si il percute son corps fin de partie
            self.init_new_game()
            return False
        self.snake_positions.insert(0, next_pos)
        if (self.apple_pos == self.snake_positions[0]).all(): #pomme mangé taille augmente
            self.snake_size += 1
            self.new_apple()
        if len(self.snake_positions) > self.snake_size: #pour faire un deplacement on reduit de 1pixel le dernier bout du serpent et on ajoute 1 pixel dans la direction

            self.snake_positions.pop()

        return True

    def draw(self):
        # Dessine le serpent et la pomme
        CELL_SIZE = self.CELL_SIZE
        pygame.draw.rect(self.window, self.APPLE_COLOR, (*self.apple_pos * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for position in self.snake_positions:
            pygame.draw.rect(self.window, self.SNAKE_COLOR, (*position * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for func in self.to_call_on_draw:
            func()

    def step_game_loop(self, allow_events):
        # Gérer les événements du jeu et mettre à jour l'état du jeu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            elif allow_events and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.turn_left()
                elif event.key == pygame.K_RIGHT:
                    self.turn_right()

        # Lance dans le bonne ordre l'affichage
        self.window.blit(self.background_image, (0, 0))
        self.draw()
        pygame.display.update()
        self.clock.tick(self.fps_display.target_fps)

    def run_loop(self):
        # Boucle principale du jeu
        while self.running:
            self.move()
            self.step_game_loop(allow_events=True)

    def quit(self):
        # Quitter le jeu et effectuer les opérations de nettoyage
        self.running = False
        pygame.quit()
        for func in self.to_call_on_game_quit:
            func()

if __name__ == '__main__':
    game = Game()
    game.run_loop()
