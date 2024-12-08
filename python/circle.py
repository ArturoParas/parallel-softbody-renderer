from math import *

from spring import Spring


class Circle:
  rad = 25
  diameter = rad * 2
  mass = 1
  g = 1000
  v_thresh = 0.00008

  def __init__(self, px_prev, py_prev, px_curr, py_curr, mass=1, tag=""):
    self.px_prev = px_prev
    self.py_prev = py_prev
    self.px_curr = px_curr
    self.py_curr = py_curr
    self.px_temp = px_curr
    self.py_temp = py_curr

    self.mass = mass

    self.tag = tag

    self.springs = []

  def get_springs(self):
    return self.springs

  def get_velocity(self):
    return self.px_curr - self.px_prev, self.py_curr - self.py_prev

  def get_gravitational_force(self):
    return 0, Circle.g*self.mass

  def get_elastic_force(self,spring):

    circle1, circle2 = spring.circle1, spring.circle2

    x1, y1 = circle1.get_temp_position()
    x2, y2 = circle2.get_temp_position()

    if circle1.tag == self.tag:
      dx = (x2 - x1)
      dy = (y2 - y1)
    else:
      dx = (x1 - x2)
      dy = (y1 - y2)

    spring_length = sqrt(dx * dx + dy * dy)

    cos_theta = dx/spring_length
    sin_phi = dy/spring_length

    spring_force_magnitude = spring.k * (spring_length - spring.rest_length) / spring_length

    spring_force_x = abs(dx) * cos_theta * spring_force_magnitude
    spring_force_y = abs(dy) * sin_phi * spring_force_magnitude

    return spring_force_x, spring_force_y

  def get_current_position(self):
    return self.px_curr, self.py_curr

  def get_prev_position(self):
    return self.px_prev, self.py_prev

  def get_temp_position(self):
    return self.px_temp, self.py_temp

  def update_position(self, dt2):

    fx, fy = self.get_gravitational_force()

    for spring in self.springs:
      fex, fey = self.get_elastic_force(spring)
      fx, fy = fx + fex, fy + fey

    ax, ay = fx/self.mass, fy/self.mass

    vx = (self.px_curr - self.px_prev) * Spring.damping_constant
    vy = (self.py_curr - self.py_prev) * Spring.damping_constant

    if abs(vx) < Circle.v_thresh:
      vx = 0
    if abs(vy) < Circle.v_thresh:
      vy = 0

    px_next = self.px_curr + vx + ax * dt2
    py_next = self.py_curr + vy + ay * dt2
    self.px_prev = self.px_curr
    self.py_prev = self.py_curr
    self.px_curr = px_next
    self.py_curr = py_next

  def resolve_collision(self, circle):
    px_diff = circle.px_curr - self.px_curr
    py_diff = circle.py_curr - self.py_curr
    dist = (px_diff * px_diff + py_diff * py_diff) ** 0.5
    overlap = dist - Circle.diameter
    if overlap < 0:
      move_amt = overlap * 0.5 / dist
      self.px_temp += px_diff * move_amt
      self.py_temp += py_diff * move_amt

  def update_position_resolved(self):
    self.px_curr = self.px_temp
    self.py_curr = self.py_temp

  def update_position_moved(self):
    self.px_temp = self.px_curr
    self.py_temp = self.py_curr

  def add_spring(self,spring):
    self.springs.append(spring)