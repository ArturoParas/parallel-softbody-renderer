import math
from circle import Circle

class Grid:
  def __init__(self, width, height, circles, collision_iters=8):
    self.width = width
    self.height = height
    self.num_rows = math.ceil(self.height / Circle.diameter) + 2
    self.num_cols = math.ceil(self.width / Circle.diameter) + 2
    self.grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    for circle in circles:
      self.insert_circle(circle)
    self.transition_grid()
    self.collision_iters = collision_iters

  def insert_circle(self, circle):
    row = int(circle.y // Circle.diameter) + 1
    col = int(circle.x // Circle.diameter) + 1
    self.next_grid[row][col].append(circle)

  def transition_grid(self):
    self.grid = self.next_grid
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]

  def bounce_circles(self, curr_circle, comp_circle, new_circle):
    separation_vec, dv = Circle.get_collision_vectors(curr_circle, comp_circle)
    Circle.move(new_circle, separation_vec)
    Circle.update_velocity(new_circle, dv)

  def separate_circles(self, curr_circle, comp_circle, new_circle):
    separation_vec = Circle.get_separation_vector(curr_circle, comp_circle)
    Circle.move(new_circle, separation_vec)

  def bounce_off_border(self, circle):
    if circle.x < Circle.rad:
      circle.x = Circle.rad
      if abs(circle.dx) >= Circle.bounce_thresh:
        circle.dx = abs(circle.dx) * Circle.restitution
      else:
        circle.dx = 0
    elif circle.x > self.width - Circle.rad:
      circle.x = self.width - Circle.rad
      if abs(circle.dx) >= Circle.bounce_thresh:
        circle.dx = -abs(circle.dx) * Circle.restitution
      else:
        circle.dx = 0
    if circle.y < Circle.rad:
      circle.y = Circle.rad
      if abs(circle.dy) >= Circle.bounce_thresh:
        circle.dy = abs(circle.dy) * Circle.restitution
      else:
        circle.dy = 0
    elif circle.y > self.height - Circle.rad:
      circle.y = self.height - Circle.rad
      if abs(circle.dy) >= Circle.bounce_thresh:
        circle.dy = -abs(circle.dy) * Circle.restitution
      else:
        circle.dy = 0

  def separate_from_border(self, circle):
    circle.x = min(max(circle.x, Circle.rad), self.width - Circle.rad)
    circle.y = min(max(circle.y, Circle.rad), self.height - Circle.rad)

  def bounce_cell(self, row, col):
    TL = self.grid[row - 1][col - 1]
    TM = self.grid[row - 1][col]
    TR = self.grid[row - 1][col + 1]
    ML = self.grid[row][col - 1]
    MM = self.grid[row][col]
    MR = self.grid[row][col + 1]
    BL = self.grid[row + 1][col - 1]
    BM = self.grid[row + 1][col]
    BR = self.grid[row + 1][col + 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.bounce_circles(curr_circle, comp_circle, new_circle)
      self.bounce_off_border(new_circle)
      self.insert_circle(new_circle)

  def separate_cell(self, row, col):
    TL = self.grid[row - 1][col - 1]
    TM = self.grid[row - 1][col]
    TR = self.grid[row - 1][col + 1]
    ML = self.grid[row][col - 1]
    MM = self.grid[row][col]
    MR = self.grid[row][col + 1]
    BL = self.grid[row + 1][col - 1]
    BM = self.grid[row + 1][col]
    BR = self.grid[row + 1][col + 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      self.separate_from_border(new_circle)
      self.insert_circle(new_circle)

  def resolve_collisions(self):
    for row in range(1, self.num_rows - 1):
      for col in range(1, self.num_cols - 1):
        self.bounce_cell(row, col)
    self.transition_grid()

    for _ in range(self.collision_iters - 1):
      for row in range(1, self.num_rows - 1):
        for col in range(1, self.num_cols - 1):
          self.separate_cell(row, col)
      self.transition_grid()

  def apply_forces(self):
    # TODO: Test for numerical instability, implement Runge Kutta if there is any
    force = [0, Circle.g * Circle.mass]
    for row in self.grid:
      for cell in row:
        for curr_circle in cell:
          new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
          Circle.apply_force(new_circle, force)
          self.bounce_off_border(new_circle)
          self.insert_circle(new_circle)
    self.transition_grid()
  
  def update(self):
    self.apply_forces()
    self.resolve_collisions()
