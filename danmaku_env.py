# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pygame
from gym import spaces
from pygame import _numpysurfarray as npsurfarray

FPS = 30

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 600

STATE_WIDTH = 256
STATE_HEIGHT = 256
STATE_N_FRAMES = 4

MAX_N_BULLETS = 1000

# RGB
COLORS = {
    'back': (0, 0, 0),
    'out': (128, 128, 128),
    'message': (255, 255, 255),
    'player': (255, 255, 255),
    'enemy': (255, 255, 0),
    'bullet_r': (255, 0, 0),
    'bullet_g': (0, 255, 0),
}

SIN_RES = 360
SIN_UNIT = 2. * np.pi / SIN_RES
SIN_TABLE = np.sin(2. * np.pi * np.arange(SIN_RES, dtype='float') / SIN_RES)
COS_TABLE = np.cos(2. * np.pi * np.arange(SIN_RES, dtype='float') / SIN_RES)


def sin_table(x):
    return SIN_TABLE[int(x // SIN_UNIT) % SIN_RES]


def cos_table(x):
    return COS_TABLE[int(x // SIN_UNIT) % SIN_RES]


class StgObj(object):
    def __init__(self, pos, radius, color):
        super(StgObj, self).__init__()
        self.pos = np.asarray(pos, dtype=np.float).copy()
        self.radius = radius
        self.color = color

    def step(self):
        pass

    def draw(self, surface):
        x = int(self.pos[0] + 0.5)
        y = int(self.pos[1] + 0.5)

        pygame.draw.circle(surface, self.color, (x, y), self.radius)


class Bullet(StgObj):
    def __init__(self, pos, radius=8, velocity=(0, 0), color=COLORS['enemy']):
        super(Bullet, self).__init__(pos, radius, color)
        self.velocity = np.asarray(velocity, dtype=np.float).copy()

    def step(self):
        self.pos += self.velocity


class Enemy(StgObj):
    def __init__(self, pos, radius=4, color=COLORS['enemy'],
                 bullet_clock_wise=False, random_seed=None):
        super(Enemy, self).__init__(pos, radius, color)
        self.bullet_clock_wise = bullet_clock_wise

        self.random_state = np.random.RandomState(seed=random_seed)
        self.bullet_g_angle = 2 * np.pi * self.random_state.random_sample()
        self.bullet_r_angle = 2 * np.pi * self.random_state.random_sample()

    def step(self, width=1):
        self.pos[0] += width * (self.random_state.randint(0, 3) - 1)
        self.pos[1] += width * (self.random_state.randint(0, 3) - 1)

        self.pos[0] = min(max(0, self.pos[0]), WINDOW_WIDTH - 1)
        self.pos[1] = min(max(0, self.pos[1]), WINDOW_HEIGHT / 2)

    def generate_bullets_g(self, radius=8, speed=6, n_lines=5,
                           color=COLORS['bullet_g']):
        bullets = [
            Bullet(self.pos, radius,
                   (speed * cos_table(self.bullet_g_angle +
                                      add_angle * 2 * np.pi / n_lines),
                    -speed * sin_table(self.bullet_g_angle +
                                       add_angle * 2 * np.pi / n_lines)),
                   color)
            for add_angle in range(n_lines)
        ]

        if self.bullet_clock_wise:
            self.bullet_g_angle -= 2 * np.pi / (FPS * 2)
            if self.bullet_g_angle < 0:
                self.bullet_g_angle += 2 * np.pi
        else:
            self.bullet_g_angle += 2 * np.pi / (FPS * 2)
            if self.bullet_g_angle >= 2 * np.pi:
                self.bullet_g_angle -= 2 * np.pi

        return bullets

    def generate_bullets_r(self, radius=8, speed=6,
                           color=COLORS['bullet_r'], hard=False):
        bullets = [
            Bullet(self.pos, radius,
                   (speed * cos_table(self.bullet_g_angle),
                    -speed * sin_table(self.bullet_g_angle)),
                   color),
            Bullet(self.pos, radius,
                   (speed * cos_table(self.bullet_g_angle),
                    speed * sin_table(self.bullet_g_angle)),
                   color),
            Bullet(self.pos, radius,
                   (speed * cos_table(self.bullet_g_angle + np.pi),
                    -speed * sin_table(self.bullet_g_angle + np.pi)),
                   color),
            Bullet(self.pos, radius,
                   (speed * cos_table(self.bullet_g_angle + np.pi),
                    speed * sin_table(self.bullet_g_angle + np.pi)),
                   color),
        ]

        if self.bullet_clock_wise:
            self.bullet_g_angle -= 2 * np.pi / (FPS * 2)
            if self.bullet_g_angle < 0:
                self.bullet_g_angle += 2 * np.pi
        else:
            self.bullet_g_angle += 2 * np.pi / (FPS * 2)
            if self.bullet_g_angle >= 2 * np.pi:
                self.bullet_g_angle -= 2 * np.pi

        return bullets


class Player(StgObj):
    def __init__(self, pos, radius=4, color=COLORS['player']):
        super(Player, self).__init__(pos, radius, color)

    def step(self, action, width=1):
        if action == 6 or action == 7 or action == 8:
            self.pos[0] -= width
        if action == 2 or action == 3 or action == 4:
            self.pos[0] += width
        if action == 1 or action == 2 or action == 8:
            self.pos[1] -= width
        if action == 4 or action == 5 or action == 6:
            self.pos[1] += width

        self.pos[0] = min(max(0, self.pos[0]), WINDOW_WIDTH - 1)
        self.pos[1] = min(max(0, self.pos[1]), WINDOW_HEIGHT - 1)


class DanmakuEnv(object):
    def __init__(self, hard=False, random_seed=None):
        super(DanmakuEnv, self).__init__()
        pygame.init()
        self.surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.surface.fill(COLORS['back'])
        self.surface_image = npsurfarray.array3d(self.surface)
        self.observation_space = spaces.Box(
            -1., 1., (STATE_N_FRAMES, STATE_HEIGHT, STATE_WIDTH),
            dtype='float32')
        self.action_space = spaces.Discrete(9)
        self.random_state = np.random.RandomState(seed=random_seed)

        if hard:
            self.hard = True
            self.bullet_gen_interval = 10
            self.bullet_speed = 6
            self.bullet_g_lines = 10
        else:
            self.hard = False
            self.bullet_gen_interval = 10
            self.bullet_speed = 6
            self.bullet_g_lines = 5

    def reset(self):
        self.player = Player((WINDOW_WIDTH / 2, 3 * WINDOW_HEIGHT / 4))
        self.enemies = [
            Enemy((WINDOW_WIDTH / 3, WINDOW_HEIGHT / 3),
                  random_seed=self.random_state.randint(
                      np.iinfo(np.uint32).max)
                  ),
            Enemy((2 * WINDOW_WIDTH / 3, WINDOW_HEIGHT / 3),
                  random_seed=self.random_state.randint(
                      np.iinfo(np.uint32).max))]
        self.bullets = []

        self.surface.fill(COLORS['back'])
        self.player.draw(self.surface)
        for enemy in self.enemies:
            enemy.draw(self.surface)
        for bullet in self.bullets:
            bullet.draw(self.surface)

        self.surface_image = npsurfarray.array3d(self.surface)

        gray_surface = cv2.cvtColor(self.surface_image, cv2.COLOR_RGB2GRAY)
        st_frame = np.full((STATE_WIDTH, STATE_HEIGHT),
                           fill_value=COLORS['out'][0], dtype='uint8')

        pt_x_int = int(self.player.pos[0] + 0.5)
        pt_y_int = int(self.player.pos[1] + 0.5)

        st_ltrb = (max(0, pt_x_int - STATE_WIDTH // 2),
                   max(0, pt_y_int - STATE_HEIGHT // 2),
                   min(pt_x_int + STATE_WIDTH // 2, WINDOW_WIDTH),
                   min(pt_y_int + STATE_HEIGHT // 2, WINDOW_HEIGHT))

        st_frame[
            st_ltrb[0] - pt_x_int + STATE_WIDTH // 2:
            st_ltrb[2] - pt_x_int + STATE_WIDTH // 2,
            st_ltrb[1] - pt_y_int + STATE_HEIGHT // 2:
            st_ltrb[3] - pt_y_int + STATE_HEIGHT // 2,
        ] = gray_surface[st_ltrb[0]:st_ltrb[2], st_ltrb[1]:st_ltrb[3]]

        st_frame = st_frame.transpose(1, 0).astype(np.float32) / 127.5 - 1.

        self.state = np.stack(
            [st_frame for _ in range(STATE_N_FRAMES)], axis=0)
        self.reward = 0.
        self.collision = False
        self.t = 0

        return self.state

    def step(self, action):
        self.surface.fill(COLORS['back'])

        self.player.step(action, width=4)
        self.player.draw(self.surface)

        # left, top, right, bottom
        vic_ltrb = (
            max(0, int(self.player.pos[0] + 0.5) - self.player.radius * 2),
            max(0, int(self.player.pos[1] + 0.5) - self.player.radius * 2),
            min(int(self.player.pos[0] + 0.5) + self.player.radius * 2,
                WINDOW_WIDTH),
            min(int(self.player.pos[1] + 0.5) + self.player.radius * 2,
                WINDOW_HEIGHT)
        )

        vicinity = npsurfarray.array3d(
            self.surface)[vic_ltrb[0]:vic_ltrb[2], vic_ltrb[1]:vic_ltrb[3]]
        player_region = (vicinity == COLORS['player']).all(axis=2)

        for enemy in self.enemies:
            enemy.step()
            enemy.draw(self.surface)
            if self.t % self.bullet_gen_interval == 0:
                self.bullets += enemy.generate_bullets_g(
                    speed=self.bullet_speed, n_lines=self.bullet_g_lines
                )
                self.bullets += enemy.generate_bullets_r(
                    speed=self.bullet_speed, hard=self.hard,
                )

        if len(self.bullets) > MAX_N_BULLETS:
            self.bullets = self.bullets[-MAX_N_BULLETS:]

        for bullet in self.bullets:
            bullet.step()
            bullet.draw(self.surface)

        self.surface_image = npsurfarray.array3d(self.surface)
        vicinity = self.surface_image[
            vic_ltrb[0]:vic_ltrb[2], vic_ltrb[1]:vic_ltrb[3]]
        danger_region = (vicinity == COLORS['enemy']).all(axis=2)
        danger_region |= (vicinity == COLORS['bullet_g']).all(axis=2)
        danger_region |= (vicinity == COLORS['bullet_r']).all(axis=2)

        gray_surface = cv2.cvtColor(self.surface_image, cv2.COLOR_RGB2GRAY)
        st_frame = np.full((STATE_WIDTH, STATE_HEIGHT),
                           fill_value=COLORS['out'][0], dtype='uint8')

        pt_x_int = int(self.player.pos[0] + 0.5)
        pt_y_int = int(self.player.pos[1] + 0.5)

        st_ltrb = (max(0, pt_x_int - STATE_WIDTH // 2),
                   max(0, pt_y_int - STATE_HEIGHT // 2),
                   min(pt_x_int + STATE_WIDTH // 2, WINDOW_WIDTH),
                   min(pt_y_int + STATE_HEIGHT // 2, WINDOW_HEIGHT))

        st_frame[
            st_ltrb[0] - pt_x_int + STATE_WIDTH // 2:
            st_ltrb[2] - pt_x_int + STATE_WIDTH // 2,
            st_ltrb[1] - pt_y_int + STATE_HEIGHT // 2:
            st_ltrb[3] - pt_y_int + STATE_HEIGHT // 2,
        ] = gray_surface[st_ltrb[0]:st_ltrb[2], st_ltrb[1]:st_ltrb[3]]

        st_frame = st_frame.transpose(
            1, 0)[np.newaxis, :, :].astype(np.float32) / 127.5 - 1.

        self.state = np.concatenate((self.state[1:], st_frame), axis=0)

        self.collision = (player_region & danger_region).any()

        self.reward = 0.
        if self.collision:
            self.reward = -1.

        self.t += 1

        return self.state, self.reward, self.collision, {}

    def render(self):
        return self.surface_image.transpose(1, 0, 2)
