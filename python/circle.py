class Circle:
  # Static vars
  rad = 25
  diameter = rad * 2

  def __init__(self, x, y):
    self.x = x
    self.y = y

  @staticmethod
  def get_collision_vector(circle1, circle2):
    vec = [circle2.x - circle1.x, circle2.y - circle1.y]
    dist = 0
    for elem in vec:
      dist += elem ** 2
    dist = dist ** 0.5
    overlap = dist - Circle.diameter
    if overlap < 0:
      move_amt = overlap / 2
      vec = [(vec[0] / dist) * move_amt, (vec[1] / dist) * move_amt]
      return vec
    return [0, 0]

  @staticmethod
  def move(circle, vec):
    circle.x += vec[0]
    circle.y += vec[1]
