import pygame as pg
from sys import exit
from time import sleep

from utils import *
from circle import Circle
from solver import Solver


pg.init()

surface = pg.display.set_mode((0, 0))
width, height = pg.display.get_surface().get_size()
height -= 80

solver = Solver(width, height, [],[])
circles,springs = read_softbody_file("../tests/threepoint.txt")

for circle in circles:
  solver.insert_circle(circle)

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

  for c1_position, c2_position in solver.get_spring_draw_buffer():
    pg.draw.line(surface,(0,255,0),c1_position,c2_position,width=2)
  for x,y in solver.get_circle_draw_buffer():
    pg.draw.circle(surface, (0, 0, 255), (x, y), Circle.rad)


  pg.display.update()

  sleep(solver.dt)
  solver.update()

