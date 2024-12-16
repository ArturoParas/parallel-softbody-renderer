import pygame as pg
import sys
from time import sleep

pg.init()

screen = pg.display.set_mode((0, 0))
w, h = pg.display.get_surface().get_size()
print(w)
print(h)

circle_pos = [50, 50]
circle_rad = 25

while True:

  for event in pg.event.get():
    if event.type == pg.QUIT:
      pg.quit()
      sys.exit()

    if event.type == pg.KEYDOWN:
      key = event.key
      if key == pg.K_ESCAPE:
        pg.quit()
        sys.exit()

  screen.fill((0, 0, 0))
  pg.draw.circle(screen, (255, 0, 0), circle_pos, circle_rad)
  pg.display.update()
