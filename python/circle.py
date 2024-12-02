class Circle:
  rad = 25
  diameter = rad * 2
  mass = 1
  g = 0.5
  restitution = 0.5
  norm_restitution = 0.5 * restitution + 0.5
  bounce_thresh = 1

  def __init__(self, x, y, dx=0, dy=0):
    self.x = x
    self.y = y
    self.dx = dx
    self.dy = dy

  @staticmethod
  def get_collision_vectors(circle1, circle2):
    pos_diff = [circle2.x - circle1.x, circle2.y - circle1.y]
    dist = 0
    for diff in pos_diff:
      dist += diff ** 2
    dist = dist ** 0.5

    separation_vec = [0, 0]
    vel_norm = [0, 0]
    overlap = dist - Circle.diameter
    if overlap < 0:
      move_amt = overlap / 2
      pos_norm = [pos_diff[0] / dist, pos_diff[1] / dist]
      separation_vec = [pos_norm[0] * move_amt, pos_norm[1] * move_amt]
      vel_rel = (circle2.dx - circle1.dx) * pos_norm[0] + (circle2.dy - circle1.dy) * pos_norm[1]
      if abs(vel_rel) >= Circle.bounce_thresh:
        vel_norm = [vel_rel * pos_norm[0] * Circle.norm_restitution, vel_rel * pos_norm[1] * Circle.norm_restitution]
      else:
        vel_norm = [-circle1.dx, -circle2.dx]

    return separation_vec, vel_norm

  @staticmethod
  def get_separation_vector(circle1, circle2):
    pos_diff = [circle2.x - circle1.x, circle2.y - circle1.y]
    dist = 0
    for diff in pos_diff:
      dist += diff ** 2
    dist = dist ** 0.5

    separation_vec = [0, 0]
    overlap = dist - Circle.diameter
    if overlap < 0:
      move_amt = overlap / 2
      separation_vec = [pos_diff[0] / dist * move_amt, pos_diff[1] / dist * move_amt]

    return separation_vec

  @staticmethod
  def apply_force(circle, force):
    circle.dx += force[0] / Circle.mass
    circle.dy += force[1] / Circle.mass
    circle.x += circle.dx
    circle.y += circle.dy

  @staticmethod
  def move(circle, dpos):
    circle.x += dpos[0]
    circle.y += dpos[1]

  @staticmethod
  def update_velocity(circle, dvel):
    circle.dx += dvel[0]
    circle.dy += dvel[1]
