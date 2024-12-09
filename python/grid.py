import math
from circle import Circle

class Grid:
  def __init__(self, width, height, circles):
    num_rows = math.ceil(height / Circle.diameter)
    num_cols = math.ceil(width / Circle.diameter)
    self.grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
    self.next_grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
    for circle in circles:
      self.insert_circle(circle)

  def insert_circle(self, circle):
    row = circle.x // Circle.diameter
    col = circle.y // Circle.diameter
    self.grid[row][col].append(circle)

  def correct_cell_collisions(self, cell1, cell2):
    corrected_circles = []
    for _ in range(len(cell1)):
      circle1 = cell1.pop()
      for _ in range(len(cell2)):
        circle2 = cell2.pop()
        Circle.correct_collision(circle1, circle2)
        corrected_circles.append(circle2)
      corrected_circles.append(circle1)

    for circle in corrected_circles:
      self.insert_circle(circle)

  def correct_within_cell_collisions(self, cell):
    pass

  def correct_collisions(self):
    # TODO: correct collisions on top left borders

    for row in range(1, len(self.grid) - 1):
      for col in range(1, len(self.grid[row]) - 1):
        curr_cell = self.grid[row][col]
        if curr_cell:
          comp_cell = self.grid[row - 1][col - 1]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row - 1][col]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row - 1][col + 1]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row][col - 1]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row][col]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row][col + 1]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row + 1][col - 1]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row + 1][col]
          self.correct_cell_collisions(curr_cell, comp_cell)
          comp_cell = self.grid[row + 1][col + 1]
          self.correct_cell_collisions(curr_cell, comp_cell)

    # TODO: correct collisions on bottom right borders
