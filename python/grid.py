import math
from circle import Circle

class Grid:
  def __init__(self, width, height, circles):
    self.num_rows = math.ceil(height / Circle.diameter)
    self.num_cols = math.ceil(width / Circle.diameter)
    self.grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    for circle in circles:
      self.insert_circle(circle)
    self.update_grid()

  def insert_circle(self, circle):
    row = int(circle.x // Circle.diameter)
    col = int(circle.y // Circle.diameter)
    self.next_grid[row][col].append(circle)

  def update_grid(self):
    self.grid = self.next_grid
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]

  def correct_cell_collisions(self, row, col):
    curr_cell = self.grid[row][col]
    if not curr_cell:
      return

    TL = self.grid[row - 1][col - 1]
    TM = self.grid[row - 1][col]
    TR = self.grid[row - 1][col + 1]
    ML = self.grid[row][col - 1]
    MM = self.grid[row][col]
    MR = self.grid[row][col + 1]
    BL = self.grid[row + 1][col - 1]
    BM = self.grid[row + 1][col]
    BR = self.grid[row + 1][col + 1]
    for curr_circle in curr_cell:
      new_circle = Circle(curr_circle.x, curr_circle.y)
      for comp_circle in TL:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in TM:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in TR:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in ML:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in MM:
        if curr_circle.x == comp_circle.x and curr_circle.y == comp_circle.y:
          continue
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in MR:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in BL:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in BM:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      for comp_circle in BR:
        vec = Circle.get_collision_vector(curr_circle, comp_circle)
        Circle.move(new_circle, vec)
      self.insert_circle(new_circle)

  def correct_collisions(self):
    # TODO: Correct collisions on top / left borders

    for _ in range(8):
      for row in range(1, self.num_rows - 1):
        for col in range(1, self.num_cols - 1):
          self.correct_cell_collisions(row, col)
      self.update_grid()

    # TODO: Correct collisions on bottom / right borders
