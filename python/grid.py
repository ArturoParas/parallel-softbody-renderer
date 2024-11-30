import math
from circle import Circle

class Grid:
  def __init__(self, width, height, circles, collision_iters=8):
    self.num_rows = math.ceil(height / Circle.diameter)
    self.num_cols = math.ceil(width / Circle.diameter)
    self.grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    for circle in circles:
      self.insert_circle(circle)
    self.transition_grid()
    self.collision_iters = collision_iters

  def insert_circle(self, circle):
    row = int(circle.y // Circle.diameter)
    col = int(circle.x // Circle.diameter)
    if 0 <= row < self.num_rows and 0 <= col < self.num_cols:
      self.next_grid[row][col].append(circle)
    else:
      print("A circle left the screen!")

  def transition_grid(self):
    self.grid = self.next_grid
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]

  def separate_and_deflect_circles(self, curr_circle, comp_circle, new_circle):
    separation_vec, dv = Circle.get_collision_vectors(curr_circle, comp_circle)
    Circle.move(new_circle, separation_vec)
    Circle.update_velocity(new_circle, dv)

  def separate_circles(self, curr_circle, comp_circle, new_circle):
    separation_vec = Circle.get_separation_vector(curr_circle, comp_circle)
    Circle.move(new_circle, separation_vec)

  def separate_and_deflect_TL_cell(self):
    MM = self.grid[0][0]
    MR = self.grid[0][1]
    BM = self.grid[1][0]
    BR = self.grid[1][1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_TR_cell(self):
    ML = self.grid[0][self.num_cols - 2]
    MM = self.grid[0][self.num_cols - 1]
    BL = self.grid[1][self.num_cols - 2]
    BM = self.grid[1][self.num_cols - 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in ML:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_BL_cell(self):
    TM = self.grid[self.num_rows - 2][0]
    TR = self.grid[self.num_rows - 2][1]
    MM = self.grid[self.num_rows - 1][0]
    MR = self.grid[self.num_rows - 1][1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_BR_cell(self):
    TL = self.grid[self.num_rows - 2][self.num_cols - 2]
    TM = self.grid[self.num_rows - 2][self.num_cols - 1]
    ML = self.grid[self.num_rows - 1][self.num_cols - 2]
    MM = self.grid[self.num_rows - 1][self.num_cols - 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_T_cell(self, col):
    ML = self.grid[0][col - 1]
    MM = self.grid[0][col]
    MR = self.grid[0][col + 1]
    BL = self.grid[1][col - 1]
    BM = self.grid[1][col]
    BR = self.grid[1][col + 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in ML:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_L_cell(self, row):
    TM = self.grid[row - 1][0]
    TR = self.grid[row - 1][1]
    MM = self.grid[row][0]
    MR = self.grid[row][1]
    BM = self.grid[row + 1][0]
    BR = self.grid[row + 1][1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_R_cell(self, row):
    TL = self.grid[row - 1][self.num_cols - 2]
    TM = self.grid[row - 1][self.num_cols - 1]
    ML = self.grid[row][self.num_cols - 2]
    MM = self.grid[row][self.num_cols - 1]
    BL = self.grid[row + 1][self.num_cols - 2]
    BM = self.grid[row + 1][self.num_cols - 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_B_cell(self, col):
    TL = self.grid[self.num_rows - 2][col - 1]
    TM = self.grid[self.num_rows - 2][col]
    TR = self.grid[self.num_rows - 2][col + 1]
    ML = self.grid[self.num_rows - 1][col - 1]
    MM = self.grid[self.num_rows - 1][col]
    MR = self.grid[self.num_rows - 1][col + 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_and_deflect_internal_cell(self, row, col):
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
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_and_deflect_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_TL_cell(self):
    MM = self.grid[0][0]
    MR = self.grid[0][1]
    BM = self.grid[1][0]
    BR = self.grid[1][1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_TR_cell(self):
    ML = self.grid[0][self.num_cols - 2]
    MM = self.grid[0][self.num_cols - 1]
    BL = self.grid[1][self.num_cols - 2]
    BM = self.grid[1][self.num_cols - 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in ML:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_BL_cell(self):
    TM = self.grid[self.num_rows - 2][0]
    TR = self.grid[self.num_rows - 2][1]
    MM = self.grid[self.num_rows - 1][0]
    MR = self.grid[self.num_rows - 1][1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_BR_cell(self):
    TL = self.grid[self.num_rows - 2][self.num_cols - 2]
    TM = self.grid[self.num_rows - 2][self.num_cols - 1]
    ML = self.grid[self.num_rows - 1][self.num_cols - 2]
    MM = self.grid[self.num_rows - 1][self.num_cols - 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_T_cell(self, col):
    ML = self.grid[0][col - 1]
    MM = self.grid[0][col]
    MR = self.grid[0][col + 1]
    BL = self.grid[1][col - 1]
    BM = self.grid[1][col]
    BR = self.grid[1][col + 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
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
      self.insert_circle(new_circle)

  def separate_L_cell(self, row):
    TM = self.grid[row - 1][0]
    TR = self.grid[row - 1][1]
    MM = self.grid[row][0]
    MR = self.grid[row][1]
    BM = self.grid[row + 1][0]
    BR = self.grid[row + 1][1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BR:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_R_cell(self, row):
    TL = self.grid[row - 1][self.num_cols - 2]
    TM = self.grid[row - 1][self.num_cols - 1]
    ML = self.grid[row][self.num_cols - 2]
    MM = self.grid[row][self.num_cols - 1]
    BL = self.grid[row + 1][self.num_cols - 2]
    BM = self.grid[row + 1][self.num_cols - 1]
    for curr_circle in MM:
      new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
      for comp_circle in TL:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in TM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in ML:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in MM:
        if curr_circle.x != comp_circle.x or curr_circle.y != comp_circle.y:
          self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BL:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      for comp_circle in BM:
        self.separate_circles(curr_circle, comp_circle, new_circle)
      self.insert_circle(new_circle)

  def separate_B_cell(self, col):
    TL = self.grid[self.num_rows - 2][col - 1]
    TM = self.grid[self.num_rows - 2][col]
    TR = self.grid[self.num_rows - 2][col + 1]
    ML = self.grid[self.num_rows - 1][col - 1]
    MM = self.grid[self.num_rows - 1][col]
    MR = self.grid[self.num_rows - 1][col + 1]
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
      self.insert_circle(new_circle)

  def separate_internal_cell(self, row, col):
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
      self.insert_circle(new_circle)

  def separate_and_deflect(self):
    self.separate_and_deflect_TL_cell()
    for col in range(1, self.num_cols - 1):
      self.separate_and_deflect_T_cell(col)
    self.separate_and_deflect_TR_cell()

    for row in range(1, self.num_rows - 1):
      self.separate_and_deflect_L_cell(row)
      for col in range(1, self.num_cols - 1):
        self.separate_and_deflect_internal_cell(row, col)
      self.separate_and_deflect_R_cell(row)

    self.separate_and_deflect_BL_cell()
    for col in range(1, self.num_cols - 1):
      self.separate_and_deflect_B_cell(col)
    self.separate_and_deflect_BR_cell()

  def separate(self):
    self.separate_TL_cell()
    for col in range(1, self.num_cols - 1):
      self.separate_T_cell(col)
    self.separate_TR_cell()

    for row in range(1, self.num_rows - 1):
      self.separate_L_cell(row)
      for col in range(1, self.num_cols - 1):
        self.separate_internal_cell(row, col)
      self.separate_R_cell(row)

    self.separate_BL_cell()
    for col in range(1, self.num_cols - 1):
      self.separate_B_cell(col)
    self.separate_BR_cell()

  def correct_collisions(self):
    self.separate_and_deflect()
    self.transition_grid()
    for _ in range(self.collision_iters - 1):
      self.separate()
      self.transition_grid()

  def apply_forces(self):
    # TODO: Test for numerical instability, implement Runge Kutta if there is any
    force = [0, Circle.g * Circle.mass]
    for row in self.grid:
      for cell in row:
        for curr_circle in cell:
          new_circle = Circle(curr_circle.x, curr_circle.y, curr_circle.dx, curr_circle.dy)
          Circle.apply_force(new_circle, force)
          self.insert_circle(new_circle)
    self.transition_grid()
  
  def update(self):
    self.apply_forces()
    self.correct_collisions()
