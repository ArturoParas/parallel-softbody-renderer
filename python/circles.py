class Circle:
  # Static vars
  rad = 25
  diameter = rad * 2

  def __init__(self, pos):
    self.x = pos[0]
    self.y = pos[1]

  @staticmethod
  def correct_collision(circle1, circle2):
    vec = [circle2.x - circle1.x, circle2.y - circle1.y]
    dist = 0
    for elem in vec:
      dist += elem**2
    dist = dist**(0.5)
    overlap = dist - Circle.diameter
    if overlap < 0:
      move_amt = overlap / 2
      vec = [(vec[0] / dist) * move_amt, (vec[1] / dist) * move_amt]
      opp_vec = [-vec[0], -vec[1]]
      Circle.move(circle1, vec)
      Circle.move(circle2, opp_vec)

  @staticmethod
  def move(circle, vec):
    circle.x += vec[0]
    circle.y += vec[1]
