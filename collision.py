import math


def oriented_rectangle_overlap_cost(state_a, state_b, half_length, half_width):
    """Return the upstream alpha-RACER simulator's body-overlap measure.

    The center displacement is resolved in each vehicle's body frame. A
    positive contribution requires simultaneous longitudinal and lateral
    overlap in that frame; both frames are checked for differing headings.
    """
    dx = float(state_a.x - state_b.x)
    dy = float(state_a.y - state_b.y)
    overlap_cost = 0.0
    for psi in (float(state_a.psi), float(state_b.psi)):
        cos_psi = math.cos(psi)
        sin_psi = math.sin(psi)
        delta_long = dx * cos_psi + dy * sin_psi
        delta_lat = dy * cos_psi - dx * sin_psi
        long_overlap = 2.0 * float(half_length) - abs(delta_long)
        lat_overlap = 2.0 * float(half_width) - abs(delta_lat)
        if long_overlap > 0.0 and lat_overlap > 0.0:
            overlap_cost += long_overlap * lat_overlap
    return float(overlap_cost)
