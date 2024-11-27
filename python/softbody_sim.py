import pygame as pg
from sys import exit
from time import sleep

from circle import Circle
from grid import Grid

pg.init()

surface = pg.display.set_mode((0, 0))
width, height = pg.display.get_surface().get_size()

occ = Grid(width, height, [Circle(400, 400), Circle(405, 425), Circle(425, 425), Circle(370, 400), Circle(360, 390), Circle(370, 360), Circle(350, 400), Circle(351, 440), Circle(315, 403), Circle(300, 450), Circle(305, 490), Circle(340, 500), Circle(360, 480), Circle(400, 483)])

while True:

  for event in pg.event.get():
    if event.type == pg.QUIT:
      pg.quit()
      exit()

    if event.type == pg.KEYDOWN:
      key = event.key
      if key == pg.K_ESCAPE:
        pg.quit()
        exit()

  surface.fill((0, 0, 0))
  for row in occ.grid:
    for cell in row:
      for circle in cell:
        pg.draw.circle(surface, (255, 0, 0), (circle.x, circle.y), Circle.rad)

  pg.display.update()

  sleep(1)
  occ.correct_collisions()
