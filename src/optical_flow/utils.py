from scipy.spatial.distance import euclidean


def inside_circle(x, y, x_circ, y_circ, r):
    dx = abs(x - x_circ)
    dy = abs(y - y_circ)
    if dx > r:
        return False
    if dy > r:
        return False
    return True


# Get arrows: (x, y, dx, dy)
def get_arrows(blobs_prev, U, V):
    step = 5
    arrows = []
    while len(arrows) != len(blobs_prev) and step >= 1:
        step -= 2
        arrows = get_arrows_internal(blobs_prev, U, V, step)
    return arrows


def get_arrows_internal(blobs_prev, u, v, step):
    circle_arrows_list = [[] for _ in range(len(blobs_prev))]
    for i in range(0, u.shape[0], step):
        for j in range(0, v.shape[1], step):
            x, y = (j + v[i, j], i + u[i, j])
            for circle_index, (y_circ, x_circ, r) in enumerate(blobs_prev):
                if inside_circle(x, y, x_circ, y_circ, r):
                    circle_arrows_list[circle_index].append((v[i, j], u[i, j]))
    arrows = []
    for circle_index, circle_arrows in enumerate(circle_arrows_list):
        mean_x = 0
        mean_y = 0
        if len(circle_arrows) != 0:
            for (x_arrow, y_arrow) in circle_arrows:
                mean_x += x_arrow
                mean_y += y_arrow
            mean_x /= len(circle_arrows)
            mean_y /= len(circle_arrows)
            arrows.append((blobs_prev[circle_index][1], blobs_prev[circle_index][0], mean_y, mean_x))
    return arrows


def calculate_total_squared_error(arrows, linked, not_found=0):  # TODO: verslag
    total_error = 0
    total_elements = len(arrows)
    for i in range(total_elements):
        x, y, dx, dy = arrows[i]
        arrow_coords = (x + dx, y + dy)
        actual_blob = linked[i][1]
        if actual_blob is not None:
            blob_coords = int(actual_blob[1]), int(actual_blob[0])
            total_error += euclidean(arrow_coords, blob_coords) ** 2
        else:
            not_found += 1
    assert total_elements != 0, "No movement detected!"
    return total_error * ((1 + not_found) / total_elements)
