#!/usr/bin/env python

import sys
import pygame
import argparse
import numpy as np
from itertools import *

class BallModel:
    DRAG = 0.995

    def __init__(self, start_position, radius):
        self.position = start_position
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        self.xdot += delta_xdot/5.0
        self.ydot += delta_ydot/5.0
        self.__clip(self.xdot)
        self.__clip(self.ydot)

    def add_drag(self):
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        self.position[0] += self.xdot*self.radius/20.0
        self.position[1] += self.ydot*self.radius/20.0

    def __clip(self, val, low=-1, high=1):
        if val > high:
            val = high
        if val < low:
            val = low
        return val

class PinballObstacle:
    def __init__(self, points):
        self.points = points
        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = min(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None

    def collision(self, ball):
        """
        Determine if the ball hits this obstacle
        """
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.array(self.points))
        next(b, None)
        intercept_found = False
        for pt_pair in izip(a, b):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(pt_pair, intercept, ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    self.intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        """
        Based of the collision detection result triggered
        in L{PinballObstacle.collision()}, compute the
        change in velocity.
        """
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        # Normalize direction
        if self._intercept[0] < 0:
            self._intercept[:] = self._intercept*-1

        angle = self._angle(self._intercept[:], [ball.xdot, ball.ydot])
        angle -= np.pi

        intercept_theta = self._angle([-1, 0], self._intercept)
        angle += intercept_theta

        if angle > 2*np.pi:
            angle -= 2*np.pi

        ball_velocity = np.linalg.norm([ball.xdot, ball.ydot])
        return [ball_velocity*np.cos(angle), ball_velocity*np.sin(angle)]

    def _angle(self, v1, v2):
        return np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

    def _select_edge(self, edge1, edge2, ball):
        """
        If the ball hits a corner, select one of two edges.
        @return the edge with the smallest angle with the velocity vector
        """
        velocity = np.array(ball.xdot, ball.ydot)
        angle1 = self._angle(velocity, edge1)
        angle2 = self._angle(velocity, edge2)

        if angle1 > np.pi:
            angle1 -= np.pi
        if angle1 > np.pi:
            angle2 -= np.pi

        if np.abs(angle1 - (np.pi/2.0)) < np.abs(angle2 - (np.pi/2.0)):
            return edge1
        return edge2

    def _intercept_edge(self, pt_pair, ball):
        """
        @return True if the ball has hit an edge of the polygon
        """
        # Find the projection on an edge
        direction = pt_pair[1] - pt_pair[0]
        difference = pt_pair[1] - ball.position

        scalar_proj = difference.dot(direction)/difference.dot(difference)
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        # Compute the distance to the closest point
        closest_pt = pt_pair[1] + direction*scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance < ball.radius*ball.radius:
            # A collision only if the ball is not already moving away
            ball_to_obstacle  = closest_pt - ball.position
            velocity = np.array([ball.xdot, ball.ydot])
            angle = self._angle(velocity, ball_to_obstacle)
            if angle > np.pi:
                angle = 2*np.pi - angle

            if angle > np.pi/1.99:
                return False
            return True
        else:
            return False


class PinballModel:
    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4

    STEP_PENALTY = -1
    THRUST_PENALTY = -5
    END_EPISODE = 10000

    def __init__(self, configuration):
        """
        Read a configuration file for Pinball and draw the domain to screen
        @param configuration: a configuration file containing the polygons,
        source and target locations.
        """
        self.action_effects = {self.ACC_X:(1, 0), self.ACC_Y:(0, 1), self.DEC_X:(-1, 0), self.DEC_Y:(0, -1), self.ACC_NONE:(0, 0)}


        # Set up the environment according to the configuration
        self.obstacles = []
        self.target_pos = []
        self.target_rad = 0.01

        start_pos = 0.01
        ball_rad = 0.01
        with open(configuration) as fp:
            for line in fp.readlines():
                tokens = line.strip().split()
                if not len(tokens):
                    continue
                elif tokens[0] == 'polygon':
                    self.obstacles.append(
                        PinballObstacle(zip(*[iter(map(float, tokens[1:]))] * 2)))
                elif tokens[0] == 'target':
                    self.target_pos = [float(tokens[1]), float(tokens[2])]
                    self.target_rad = float(tokens[3])
                elif tokens[0] == 'start':
                    start_pos = [float(tokens[1]), float(tokens[2])]
                elif tokens[0] == 'ball':
                    ball_rad = float(tokens[1])

        self.ball = BallModel(start_pos, ball_rad)

    def take_action(self, action):

        for i in xrange(20):
            if i == 0:
                self.ball.add_impulse(*self.action_effects[action])
            self.ball.step()

            # Detect collisions
            ncollision = 0
            dxdy = np.array([0, 0])

            for obs in self.obstacles:
                if obs.collision(self.ball):
                    dxdy += obs.collision_effect(ball)
                    ncollision += 1

            if ncollision == 1:
                self.ball.xdot = dxdy[0]
                self.ball.ydot = dxdy[1]
                if i == 19:
                    self.ball.step()
            elif ncollision > 1:
                self.ball.xdot = -self.ball.xdot
                self.ball.ydot = -self.ball.ydot

            if self.episode_ended():
                return self.END_EPISODE

        self.ball.add_drag()
        self._check_bounds()

        if action == self.ACC_NONE:
            return self.STEP_PENALTY

        return self.THRUST_PENALTY

    def episode_ended(self):
        return np.linalg.norm(np.array(self.ball.position)-np.array(self.target_pos)) < self.target_rad

    def _check_bounds(self):
        if self.ball.position[0] > 1.0:
            self.ball.position[0] = 0.95
        if self.ball.position[0] < 0.0:
            self.ball.position[0] = 0.05
        if self.ball.position[1] > 1.0:
            self.ball.position[1] = 0.95
        if self.ball.position[1] < 0.0:
            self.ball.position[1] = 0.05


class PinballView:
    def __init__(self, screen, model):
        self.screen = screen
        self.model = model

        self.DARK_GRAY = [64, 64, 64]
        self.LIGHT_GRAY = [232, 232, 232]
        self.BALL_COLOR = [0, 0, 255]
        self.TARGET_COLOR = [255, 0, 0]

        # Draw the background
        self.background_surface = pygame.Surface(screen.get_size())
        self.background_surface.fill(self.LIGHT_GRAY)
        for obs in model.obstacles:
            pygame.draw.polygon(self.background_surface, self.DARK_GRAY, map(self.__to_pixels, obs.points), 0)

        pygame.draw.circle(
            screen, self.TARGET_COLOR, self.__to_pixels(self.model.target_pos), int(self.model.target_rad*self.screen.get_width()))

    def __to_pixels(self, pt):
         return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

    def blit(self):
        self.screen.blit(self.background_surface, (0, 0))
        pygame.draw.circle(self.screen, self.BALL_COLOR,
                           self.__to_pixels(self.model.ball.position), int(self.model.ball.radius*self.screen.get_width()))


def run_game():
    parser = argparse.ArgumentParser(description='Pinball domain')
    parser.add_argument('configuration', help='The configuration file')
    parser.add_argument('--width', action='store', type=int,
                        default=500, help='screen width (default: 500)')
    parser.add_argument('--height', action='store', type=int,
                        default=500, help='screen height (default: 500)')
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode([args.width, args.height])

    environment = PinballModel(args.configuration)
    environment_view = PinballView(screen, environment)

    actions = {pygame.K_RIGHT:PinballModel.ACC_X, pygame.K_UP:PinballModel.ACC_Y, pygame.K_LEFT:PinballModel.DEC_X, pygame.K_DOWN:PinballModel.DEC_Y}

    done = False
    while not done:
        pygame.time.wait(100)

        user_action = PinballModel.ACC_NONE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
                user_action = actions.get(event.key, PinballModel.ACC_NONE)

        environment.take_action(user_action)
        environment_view.blit()

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    run_game()
