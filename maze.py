import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 600, 600
TILE_SIZE = 40
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)

# Maze layout: 1 = wall, 0 = path, G = goal

maze = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,1,0,0,0,0,1,0,0,0,1,0,0,1],
    [1,0,1,0,1,1,0,1,0,1,0,1,1,0,1],
    [1,0,1,0,1,0,0,0,0,1,0,0,1,0,1],
    [1,0,1,0,1,1,1,1,0,1,1,0,1,0,1],
    [1,0,0,0,0,0,0,1,0,0,0,0,1,0,1],
    [1,1,1,1,1,1,0,1,1,1,1,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,1,1,1,1,1,0,1],
    [1,0,0,1,0,0,0,0,0,0,0,0,0,'G',1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]

# Player
player_pos = [1, 1]  # Row, Col

def draw_maze():
    screen.fill(BLACK)
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            x = col * TILE_SIZE
            y = row * TILE_SIZE
            if maze[row][col] == 1:
                pygame.draw.rect(screen, WHITE, (x, y, TILE_SIZE, TILE_SIZE))
            elif maze[row][col] == 'G':
                pygame.draw.rect(screen, GREEN, (x, y, TILE_SIZE, TILE_SIZE))

    # Draw player
    px, py = player_pos[1] * TILE_SIZE, player_pos[0] * TILE_SIZE
    pygame.draw.rect(screen, BLUE, (px, py, TILE_SIZE, TILE_SIZE))

# Game loop
clock = pygame.time.Clock()
running = True
while running:
    clock.tick(10)
    draw_maze()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player movement
    keys = pygame.key.get_pressed()
    new_pos = player_pos[:]
    if keys[pygame.K_LEFT]:
        new_pos[1] -= 1
    elif keys[pygame.K_RIGHT]:
        new_pos[1] += 1
    elif keys[pygame.K_UP]:
        new_pos[0] -= 1
    elif keys[pygame.K_DOWN]:
        new_pos[0] += 1

    if maze[new_pos[0]][new_pos[1]] != 1:
        player_pos = new_pos

    # Win condition
    if maze[player_pos[0]][player_pos[1]] == 'G':
        print("🎉 You reached the goal!")
        pygame.time.delay(2000)
        running = False

pygame.quit()
sys.exit()
