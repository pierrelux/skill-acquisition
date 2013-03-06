import pygame


class Pinball:
    """
    The PinBall domain is a 4-dimensional continuous test domain for
    Reinforcement Learning (RL) algorithms.
    """
    def __init__(self, configuration):
        """
        Read a configuration file for Pinball and draw the domain to screen
        @param configuration: a configuration file containing the polygons,
        source and target locations.
        """
        self.start = []
        self.obstacles = []
        self.target_pos = []
        self.target_rad = 0.0
        self.ball_radius = 0.01

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

    def scale(self, str_lst):
        return map(lambda tok: int(float(tok) * 500), str_lst)

# Pinball
pinball = Pinball('pinball_hard_single.cfg')

# Draw the environment on screen
DARK_GRAY = [64, 64, 64]
LIGHT_GRAY = [232, 232, 232]
TARGET_COLOR = [255, 0, 0]

pygame.init()
screen = pygame.display.set_mode([500, 500])
pygame.display.set_caption("Pinball Domain")

done = False
clock = pygame.time.Clock()
while not done:
    clock.tick(100)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    screen.fill(LIGHT_GRAY)

    # Draw obstacles
    for polygon in pinball.obstacles:
        pygame.draw.polygon(screen, DARK_GRAY, polygon, 0)

    # Draw target
    pygame.draw.circle(screen, TARGET_COLOR, pinball.target_pos, pinball.target_rad)

    pygame.display.flip()

pygame.quit()
