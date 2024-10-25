import math


def dist(P):
    return math.sqrt(P.x * P.x + P.y * P.y + P.z * P.z)


def dist_p2p(P, Q):
    x = P.x - Q.x
    y = P.y - Q.y
    z = P.z - Q.z
    return math.sqrt(x*x + y*y + z*z)


def dist_sum(landmarks):
    res = 0
    if landmarks is None:
        return -1
    for point in landmarks:
        res = res + dist(point)
    return res


def coor_cos(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return (x1*x2 + y1*y2) / (math.sqrt(x1*x1 + y1*y1) * math.sqrt(x2*x2 + y2*y2))


def coor_sin(P1, P2):
    x1, y1 = P1
    x2, y2 = P2
    return (x1*y2 - y1*x2) / (math.sqrt(x1*x1 + y1*y1) * math.sqrt(x2*x2 + y2*y2))


def point_to_segment_dist(P, A, B):
    px, py, ax, ay, bx, by = P.x, P.y, A.x, A.y, B.x, B.y
    # Вектор от A до B
    ABx = bx - ax
    ABy = by - ay
    # Вектор от A до P
    APx = px - ax
    APy = py - ay
    # Проекция AP на AB, нормализованная на длину отрезка AB
    AB_AB = ABx**2 + ABy**2  # длина AB в квадрате
    if AB_AB == 0:
        # A и B - одна и та же точка
        tmp1 = px - ax
        tmp2 = py - ay
        return math.sqrt(tmp1**2 + tmp2**2)
    t = (APx * ABx + APy * ABy) / AB_AB
    t = max(0, min(1, t))  # Ограничиваем t от 0 до 1
    # Ближайшая точка на отрезке
    closest_x = ax + t * ABx
    closest_y = ay + t * ABy
    # Возвращаем расстояние от P до ближайшей точки на отрезке
    tmp1 = px - closest_x
    tmp2 = py - closest_y
    return math.sqrt(tmp1**2 + tmp2**2)


def segment_to_segment_dist(A, B, C, D):
    distances = [
        point_to_segment_dist(A, C, D),  # A до CD
        point_to_segment_dist(B, C, D),  # B до CD
        point_to_segment_dist(C, A, B),  # C до AB
        point_to_segment_dist(D, A, B)   # D до AB
    ]

    return min(distances)
