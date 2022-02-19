from scipy.spatial.distance import euclidean


def get_arrow_destination(arrow):
    x, y, dx, dy = arrow
    return x + dx, y + dy


def blobs_distance(coords1, coords2):
    return euclidean(coords1, coords2)


def blob_to_coords(blob):
    return blob[1], blob[0]


def find_nearest_actual_blob(blobs_next, arrow, threshold):
    minimal_distance = 999999  # grote int
    best_blob = None
    for actual_next in blobs_next:
        coords_actual_next = blob_to_coords(actual_next)
        coords_arrow = get_arrow_destination(arrow)
        found_distance = blobs_distance(coords_arrow, coords_actual_next)
        if found_distance < minimal_distance:
            minimal_distance = found_distance
            best_blob = actual_next
    if blobs_distance(get_arrow_destination(arrow), blob_to_coords(best_blob)) < threshold:
        return tuple(best_blob)
    return None


def link(blobs_next, arrows, threshold=7):  # TODO: Verslag!!!
    blobs_next = [blob.tolist() for blob in blobs_next]
    sources = [(x, y) for x, y, _, _ in arrows]
    nearest_blobs = [find_nearest_actual_blob(blobs_next, arrow, threshold) for arrow in arrows]
    destinations = [(dest[1], dest[0]) if dest else None for dest in nearest_blobs]  # Swap x en y
    radii = [dest[2] if dest else None for dest in nearest_blobs]
    results = list(zip(sources, destinations, radii))
    return results
