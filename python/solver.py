import math

from circle import Circle

class Solver:
  def __init__(self, width, height, circles,springs, intermediate_steps=8, dt=0.01):
    self.width = width
    self.height = height
    self.num_rows = math.ceil(self.height / Circle.diameter) + 2
    self.num_cols = math.ceil(self.width / Circle.diameter) + 2
    self.grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]
    self.circle_draw_buffer = []
    self.spring_draw_buffer = {}
    for circle in circles:
      self.insert_circle(circle)
    self.transition_grid()
    self.intermediate_steps = intermediate_steps
    self.dt = dt
    self.dt_intermediate = dt / intermediate_steps
    self.dt2_intermediate = self.dt_intermediate * self.dt_intermediate

  def get_circle_draw_buffer(self):
    return self.circle_draw_buffer

  def get_spring_draw_buffer(self):
    return list(self.spring_draw_buffer.values())


  def insert_circle(self, circle):
    row = int(circle.py_curr // Circle.diameter) + 1
    col = int(circle.px_curr // Circle.diameter) + 1
    self.next_grid[row][col].append(circle)

  def transition_grid(self):
    self.grid = self.next_grid
    self.next_grid = [[[] for _ in range(self.num_cols)] for _ in range(self.num_rows)]

  def move_circles(self):
    for row in self.grid:
      for cell in row:
        for circle in cell:
          circle.update_position(self.dt2_intermediate)

  def resolve_cell(self, row, col):
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
      curr_circle.update_position_moved()
      for comp_circle in TL:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in TM:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in TR:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in ML:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in MM:
        if curr_circle.px_curr != comp_circle.px_curr or curr_circle.py_curr != comp_circle.py_curr:
          curr_circle.resolve_collision(comp_circle)
      for comp_circle in MR:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in BL:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in BM:
        curr_circle.resolve_collision(comp_circle)
      for comp_circle in BR:
        curr_circle.resolve_collision(comp_circle)

  def resolve_collisions(self):
    for row in range(1, self.num_rows - 1):
      for col in range(1, self.num_cols - 1):
        self.resolve_cell(row, col)

  def apply_border(self):
    for row in self.grid:
      for cell in row:
        for circle in cell:
          circle.px_temp = min(max(circle.px_temp, Circle.rad), self.width - Circle.rad)
          circle.py_temp = min(max(circle.py_temp, Circle.rad), self.height - Circle.rad)
  
  def transition_circles(self):
    for row in self.grid:
      for cell in row:
        # TODO: Is it faster to pop circles (saves space I think), or is it faster just iterate
        # through circles without popping?
        while cell:
          circle = cell.pop()
          circle.update_position_resolved()
          self.insert_circle(circle)
          self.update_circle_draw_buffer(circle)
          self.update_spring_draw_buffer(circle)

  def clear_draw_buffers(self):
    self.circle_draw_buffer.clear()
    self.spring_draw_buffer.clear()

  def update_circle_draw_buffer(self,circle):
    self.circle_draw_buffer.append((circle.px_curr,circle.py_curr))

  def update_spring_draw_buffer(self,circle):
    for spring in circle.get_springs():
      if spring.tag in self.spring_draw_buffer:
        spring_circle1, _ = self.spring_draw_buffer[spring.tag]
        self.spring_draw_buffer[spring.tag] = spring_circle1, (circle.px_curr,circle.py_curr)

      else:
        self.spring_draw_buffer[spring.tag] = (circle.px_curr, circle.py_curr), (-1,-1)

  def update(self):
    for _ in range(self.intermediate_steps):
      self.clear_draw_buffers()
      self.move_circles()
      self.resolve_collisions()
      self.apply_border()
      self.transition_circles()
      self.transition_grid()
