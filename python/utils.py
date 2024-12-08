from os.path import isfile

from circle import Circle
from spring import Spring


def read_softbody_file(filename):
    if not isfile(filename):
        print("Reader could not find file: " + filename)
        return

    f = open(filename, "r")
    entries = f.readlines()

    circles = []
    springs = []

    circle_refs = {}

    params = entries[0].split(" ")
    n = int(params[0])
    s = int(params[1])

    x_positions = entries[1].split(",")
    y_positions = entries[2].split(",")

    x_displacements = entries[3].split(",")
    y_displacements = entries[4].split(",")

    masses = entries[5].split(",")

    circle_tags = entries[6].split(",")

    circle1_tags = entries[7].split(",")
    circle2_tags = entries[8].split(",")

    resting_lengths = entries[9].split(",")
    spring_constants = entries[10].split(",")

    spring_tags = entries[11].split(",")

    for i in range(n):
        x, y = float(x_positions[i]), float(y_positions[i])
        dx, dy = float(x_displacements[i]), float(y_displacements[i])
        m = float(masses[i])
        tag = circle_tags[i].strip()

        new_circle = Circle(x, y, x+dx, y+dy, m, tag)
        circles.append(new_circle)
        circle_refs[tag] = new_circle

    for j in range(s):
        c1 = circle_refs[circle1_tags[j].strip()]
        c2 = circle_refs[circle2_tags[j].strip()]
        rest_length = float(resting_lengths[j])
        k = float(spring_constants[j])
        tag = spring_tags[j].strip()

        new_spring = Spring(c1, c2, rest_length, k,tag)
        springs.append(new_spring)
        c1.add_spring(new_spring)
        c2.add_spring(new_spring)
    return circles, springs
