class Circle:
  rad = 25
  diameter = rad * 2
  mass = 1
  g = 1000
  v_thresh = 0.001

  def __init__(self, px_prev, py_prev, px_curr, py_curr):
    self.px_prev = px_prev
    self.py_prev = py_prev
    self.px_curr = px_curr
    self.py_curr = py_curr
    self.px_resolved = px_curr
    self.py_resolved = py_curr

  def get_acceleration(self):
    return 0, Circle.g

  def update_position(self, dt2):
    ax, ay = self.get_acceleration()
    vx = self.px_curr - self.px_prev
    vy = self.py_curr - self.py_prev
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
    self.px_resolved = self.px_curr
    self.py_resolved = self.py_curr

  def resolve_collision(self, circle):
    px_diff = circle.px_curr - self.px_curr
    py_diff = circle.py_curr - self.py_curr
    dist = (px_diff * px_diff + py_diff * py_diff) ** 0.5
    overlap = dist - Circle.diameter
    if overlap < 0:
      move_amt = overlap * 0.5 / dist
      self.px_resolved += px_diff * move_amt
      self.py_resolved += py_diff * move_amt

  def update_position_resolved(self):
    self.px_curr = self.px_resolved
    self.py_curr = self.py_resolved
