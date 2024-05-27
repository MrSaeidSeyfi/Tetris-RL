import gym
from gym import spaces
import numpy as np
import random
import pygame

# Define Tetris game parameters
GRID_WIDTH = 10
GRID_HEIGHT = 20
TILE_SIZE = 30
SCREEN_WIDTH = GRID_WIDTH * TILE_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * TILE_SIZE

# Define Tetris shapes and their colors
SHAPES = [
    [[1, 1, 1], [0, 1, 0]],  # T-shape
    [[0, 1, 1], [1, 1, 0]],  # S-shape
    [[1, 1, 0], [0, 1, 1]],  # Z-shape
    [[1, 1, 1, 1]],          # I-shape
    [[1, 1], [1, 1]],        # O-shape
    [[1, 1, 1], [1, 0, 0]],  # L-shape
    [[1, 1, 1], [0, 0, 1]]   # J-shape
]

COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255)   # Cyan
]

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Left, Right, Rotate, Drop
        self.observation_space = spaces.Box(low=0, high=1, shape=(GRID_HEIGHT * GRID_WIDTH,), dtype=np.uint8)  # Flattened grid
        self.reset()

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if all(row)]
        num_lines_cleared = len(lines_to_clear)
        for i in lines_to_clear:
            self.grid = np.delete(self.grid, i, axis=0)
            self.grid = np.vstack([[0] * GRID_WIDTH, self.grid])
            self.color_grid = np.delete(self.color_grid, i, axis = 0)
            self.color_grid = np.vstack([np.zeros((1,GRID_WIDTH,3),dtype=np.uint8), self.color_grid])
        return num_lines_cleared
    
    def get_height_penalty(self):
        heights = np.argmax(self.grid, axis = 0)
        max_height = GRID_HEIGHT - np.min(heights[heights >0])
        return max_height
    
    def get_holes_penalty(self):
        holes = 0
        for x in range(GRID_WIDTH):
            column = self.grid[:,x]
            first_filled = np.argmax(column)
            if first_filled>0:
                holes+= np.sum(column[first_filled:] == 0)

        return holes
    



    def reset(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.color_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)  # Grid to store colors
        self.current_piece = self.new_piece()
        self.current_color = self.get_piece_color(self.current_piece)
        self.current_position = [0, GRID_WIDTH // 2 - len(self.current_piece[0]) // 2]
        self.done = False
        return self.grid.flatten()

    def new_piece(self):
        return random.choice(SHAPES)

    def get_piece_color(self, piece):
        index = SHAPES.index(piece)
        return COLORS[index]
    
    def step(self, action):
        if action == 0:  # Left
            self.current_position[1] -= 1
            if self.check_collision():
                self.current_position[1] += 1

        elif action == 1:  # Right
            self.current_position[1] += 1
            if self.check_collision():
                self.current_position[1] -= 1

        elif action == 2:  # Rotate
            self.current_piece = np.rot90(self.current_piece)
            if self.check_collision():
                self.current_piece = np.rot90(self.current_piece,-1) # Rotate back if collision

        elif action == 3:  # Drop
            self.current_position[0] += 1
            if self.check_collision():
                self.current_position[0] -= 1
                self.place_piece()
                self.clear_lines()
                self.current_piece = self.new_piece()
                self.current_color = self.get_piece_color(self.current_piece)
                self.current_position = [0, GRID_WIDTH//2 - len(self.current_piece[0]) // 2]
                if self.check_collision():
                    self.done = True #Game over


        self.current_position[0] += 1
        
        # Check for collision
        if self.check_collision():
            self.current_position[0] -= 1
            self.place_piece()
            lines_cleared = self.clear_lines()
            height_penalty = self.get_height_penalty()
            holes_penalty = self.get_holes_penalty()

            reward = self.calculate_reward(lines_cleared, height_penalty, holes_penalty)


            self.current_piece = self.new_piece()
            self.current_color = self.get_piece_color(self.current_piece)
            self.current_position = [0, GRID_WIDTH // 2 - len(self.current_piece[0]) // 2]
            if self.check_collision():
                self.done = True  # Game over

            return self.grid.flatten(), reward, self.done, {}
        
        reward = -1
        return self.grid.flatten(), reward, self.done, {}

    def check_collision(self, offset = (0,0)):
        off_x, off_y = offset
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    new_x = x + self.current_position[1] + off_x
                    new_y = y + self.current_position[0] + off_y
                    if new_y >= GRID_HEIGHT or new_x < 0 or new_x >= GRID_WIDTH or self.grid[new_y, new_x]:
                        return True
        return False

    def place_piece(self):
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    new_y = y + self.current_position[0]
                    new_x = x + self.current_position[1]
                    if 0 <= new_y < GRID_HEIGHT and 0 <= new_x < GRID_WIDTH:
                        self.grid[new_y, new_x] = 1 
                        self.color_grid[new_y,new_x] = self.current_color

    def calculate_reward(self, lines_cleared, height_penalty, holes_penalty):
        # return np.sum(self.grid) # its simple reward /:

        line_clear_reward = {0: 0, 1: 400, 2:1000, 3:3000, 4:12000}
        reward = line_clear_reward[lines_cleared]
        # reward -= height_penalty * 0.5
        reward -= holes_penalty * 0.5
        return reward
    



    def render(self, mode='human'):
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_WIDTH + TILE_SIZE * 2, SCREEN_HEIGHT + TILE_SIZE * 2))  # Add border width
        screen.fill((0, 0, 0))

        # Draw the border
        border_color = (255, 255, 255)
        pygame.draw.rect(screen, border_color, pygame.Rect(0, 0, SCREEN_WIDTH + TILE_SIZE * 2, TILE_SIZE))  # Top border
        pygame.draw.rect(screen, border_color, pygame.Rect(0, 0, TILE_SIZE, SCREEN_HEIGHT + TILE_SIZE * 2))  # Left border
        pygame.draw.rect(screen, border_color, pygame.Rect(SCREEN_WIDTH + TILE_SIZE, 0, TILE_SIZE, SCREEN_HEIGHT + TILE_SIZE * 2))  # Right border
        pygame.draw.rect(screen, border_color, pygame.Rect(0, SCREEN_HEIGHT + TILE_SIZE, SCREEN_WIDTH + TILE_SIZE * 2, TILE_SIZE))  # Bottom border

        # Draw the grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y, x]:
                    color = tuple(self.color_grid[y,x])
                    pygame.draw.rect(screen, color, pygame.Rect((x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        # Draw the current piece with its specific color
        for y, row in enumerate(self.current_piece):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, self.current_color, pygame.Rect((x + self.current_position[1] + 1) * TILE_SIZE, (y + self.current_position[0] + 1) * TILE_SIZE, TILE_SIZE, TILE_SIZE))

        pygame.display.flip()
    
    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = TetrisEnv()
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()
