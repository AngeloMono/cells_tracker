from math import degrees
from numpy import arctan


def get_straight_line_from_2_points(p1, p2):
    """
    Calculate the equation that passed through the p1 and p2, raise a ValueError if p1=p2
    :param p1: point1 (x, y)
    :param p2: point2 (x, y)
    :returns: "(m1)x - [(m2)y] + q = 0" equation that passes through the given points
        - m1 - -1 if line x=q
        - m2 - 0 if line x=q or 1 if line y=mx+q
        - q
    """
    if p1[0] == p2[0] and p1[1] == p2[1]:
        raise ValueError("Infinite lines")
    elif p1[0] == p2[0]:
        return -1, 0, p1[0]
    m = float(p2[1] - p1[1]) / (p2[0] - p1[0])
    q = p1[1] - (m * p1[0])
    return m, 1, q


def get_degrees_from_the_x_axis(vector_x: float, vector_y: float):
    """
    Calculate the degrees of a given vector of x and y coordinates respect to the x axis.
    :param vector_x:
    :param vector_y:
    :return: degrees to the x axis [0,365[ or None if x=y=0
    """
    if vector_x == 0 and vector_y == 0:
        return None
    elif vector_x == 0:
        return 90 if vector_y > 0 else 270
    elif vector_y == 0:
        return 0 if vector_x > 0 else 180
    else:
        deg = degrees(arctan(vector_y / vector_x))
        if vector_x < 0 and vector_y < 0:
            deg = 180 + deg
        elif vector_x < 0 and vector_y > 0:
            deg = 180 + deg
        elif vector_x > 0 and vector_y < 0:
            deg = 360 + deg
    return deg
