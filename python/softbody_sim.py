import pygame as pg
from sys import exit
from time import sleep

from circle import Circle
from grid import Grid

# TODO: Springs

pg.init()

surface = pg.display.set_mode((0, 0))
width, height = pg.display.get_surface().get_size()
height -= 80

# occ = Grid(width, height, [Circle(400, 400), Circle(405, 425), Circle(425, 425), Circle(370, 400),
#                            Circle(360, 390), Circle(370, 360), Circle(350, 400), Circle(351, 440),
#                            Circle(315, 403), Circle(300, 450), Circle(305, 490), Circle(340, 500),
#                            Circle(360, 480), Circle(400, 483), Circle(600, 400, -10, 0)])
# occ = Grid(width, height, [Circle(0, 0, 10, 0), Circle(width - 1, 1, -10, 0)])

# Try with 0 g and 0.9 to 0.96 restitution for accurate billiards!
x_center = width - 401
y_center = (height - 1) / 2
dx = 25 * (3 ** 0.5)
dy = 25
occ = Grid(width, height, [Circle(200, y_center, 50), Circle(x_center, y_center),
                           Circle(x_center, y_center - 50), Circle(x_center, y_center + 50),
                           Circle(x_center - dx, y_center - dy), Circle(x_center - dx, y_center + dy),
                           Circle(x_center + dx, y_center - dy), Circle(x_center + dx, y_center + dy),
                           Circle(x_center + dx, y_center - 3 * dy), Circle(x_center + dx, y_center + 3 * dy),
                           Circle(x_center - 2 * dx, y_center)])

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

  sleep(0.01)
  occ.update()
