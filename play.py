# -*- coding: utf-8 -*-
import argparse
import pygame
from pygame.locals import QUIT, KEYDOWN
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
from pygame.locals import K_RETURN, K_ESCAPE

from danmaku_env import FPS, WINDOW_WIDTH, WINDOW_HEIGHT, DanmakuEnv

KEY_TO_ACTION = {
    0: 0,
    1: 1,
    2: 3,
    3: 2,
    4: 5,
    5: 0,
    6: 4,
    7: 3,
    8: 7,
    9: 8,
    10: 0,
    11: 1,
    12: 6,
    13: 7,
    14: 5,
    15: 0
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hard', action='store_true',
                        help='switch to hard mode.')
    args = parser.parse_args()

    pygame.init()
    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('danmaku')
    fps_clock = pygame.time.Clock()
    sys_font = pygame.font.SysFont(None, 36)

    env = DanmakuEnv(hard=args.hard)
    env.reset()

    score = 0
    high_score = 0
    fps_count = 0

    terminated = False
    game_state = 'playing'
    # 'playing', 'over'

    while True:
        if game_state == 'playing':
            key_dir_id = 0
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[K_UP]:
                key_dir_id += 1
            if pressed_keys[K_RIGHT]:
                key_dir_id += 2
            if pressed_keys[K_DOWN]:
                key_dir_id += 4
            if pressed_keys[K_LEFT]:
                key_dir_id += 8

            _, _, collision, _ = env.step(KEY_TO_ACTION[key_dir_id])

            score += 1

            if collision:
                score -= 1
                if score > high_score:
                    high_score = score
                game_state = 'over'

            for ev in pygame.event.get():
                if ev.type == QUIT:
                    terminated = True
                elif ev.type == KEYDOWN:
                    if ev.key == K_ESCAPE:
                        if game_state == 'playing':
                            score -= 1
                            if score > high_score:
                                high_score = score
                            game_state = 'over'

            surface.blit(env.surface, (0, 0))

        elif game_state == 'over':
            surface.fill((0, 0, 0))

            mes_score = sys_font.render(
                'Score: {0: 9d}'.format(score),
                True, (255, 255, 255)
            )
            mes_score_rect = mes_score.get_rect()
            mes_score_rect.center = (200, 250)

            mes_high_score = sys_font.render(
                'High Score: {0: 9d}'.format(high_score),
                True, (255, 255, 255)
            )
            mes_high_score_rect = mes_high_score.get_rect()
            mes_high_score_rect.center = (200, 300)

            mes_menu = sys_font.render(
                'Enter to restart, Esc to exit.',
                True, (255, 255, 255)
            )
            mes_menu_rect = mes_high_score.get_rect()
            mes_menu_rect.center = (150, 550)

            surface.blit(mes_score, mes_score_rect)
            surface.blit(mes_high_score, mes_high_score_rect)
            surface.blit(mes_menu, mes_menu_rect)

            for ev in pygame.event.get():
                if ev.type == QUIT:
                    terminated = True
                elif ev.type == KEYDOWN:
                    if ev.key == K_ESCAPE:
                        terminated = True
                    elif ev.key == K_RETURN:
                        env.reset()
                        score = 0
                        game_state = 'playing'

        fps_count += 1
        fps_count %= FPS

        if terminated:
            break

        pygame.display.update()
        fps_clock.tick(FPS)


if __name__ == '__main__':
    main()
