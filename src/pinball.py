import pygame
import numpy as np


class Pinball:
    """
    The PinBall domain is a 4-dimensional continuous test domain for
    Reinforcement Learning (RL) algorithms.
    """
    def __init__(self, configuration, width=1, height=1):
        """
        Read a configuration file for Pinball and draw the domain to screen
        @param configuration: a configuration file containing the polygons,
        source and target locations.
        """
        self.obstacles = []
        self.start_pos = []
        self.target_pos = []
        self.ball_rad = 0.01
        self.target_rad = 0.01
	self.width = width
	self.height = height

        with open(configuration) as fp:
            for line in fp.readlines():
                tokens = line.strip().split()

                if not len(tokens):
                    continue
                elif tokens[0] == 'polygon':
                    self.obstacles.append(zip(*[iter(self.scale(tokens[1:]))] * 2))
                elif tokens[0] == 'target':
                    target = self.scale(tokens[1:])
                    self.target_pos = target[0:-1]
                    self.target_rad = target[-1]
                elif tokens[0] == 'start':
                    self.start_pos = self.scale(tokens[1:])
                elif tokens[0] == 'ball':
                    self.ball_rad = int(float(tokens[1])*self.width)

    def scale(self, str_lst):
        return map(lambda tok: int(float(tok)*self.width), str_lst)

# Load trajectory
dataset = []
with open('trajectory.dat') as fp:
	dataset = [map(lambda x: int(float(x)*500), line.strip().split()) for line in fp.readlines()]

# Pinball
pinball = Pinball('pinball_hard_single.cfg', 500, 500)

# Draw the environment on screen
DARK_GRAY = [64, 64, 64]
LIGHT_GRAY = [232, 232, 232]
TARGET_COLOR = [255, 0, 0]
BALL_COLOR = [0, 0, 255]

pygame.init()
screen = pygame.display.set_mode([500, 500])
pygame.display.set_caption("Pinball Domain")

# Mainloop
obs_idx = 0
done = False
while not done:
    pygame.time.wait(100)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(LIGHT_GRAY)

    # Draw obstacles
    for polygon in pinball.obstacles:
        pygame.draw.polygon(screen, DARK_GRAY, polygon, 0)

    # Draw target
    pygame.draw.circle(screen, TARGET_COLOR, pinball.target_pos, pinball.target_rad)

    # Draw ball
    if obs_idx < len(dataset):
        pygame.draw.circle(screen, BALL_COLOR, dataset[obs_idx][:2], pinball.ball_rad)
        obs_idx += 1

    pygame.display.flip()

pygame.quit()
