"""Shared factual grounding for the traffic-basis vocabulary.

This module classifies observable longitudinal relationships only.  It never
maps a relationship to a racing mode, so it can be reused both by audits and by
oracle-basis interventions without treating a hand-written policy as ground
truth.
"""

from __future__ import annotations


TRAFFIC_BASIS_LABELS = (
    "no_front_vehicle",
    "closing_on_front_vehicle",
    "front_vehicle_pulling_away",
    "front_gap_approximately_constant",
    "no_rear_vehicle",
    "rear_vehicle_closing",
    "rear_vehicle_falling_behind",
    "rear_gap_approximately_constant",
)

DYNAMIC_TRAFFIC_BASIS_LABELS = (
    "closing_on_front_vehicle",
    "front_vehicle_pulling_away",
    "rear_vehicle_closing",
    "rear_vehicle_falling_behind",
)

COUNTERFACTUAL_BASIS = {
    "closing_on_front_vehicle": "front_vehicle_pulling_away",
    "front_vehicle_pulling_away": "closing_on_front_vehicle",
    "rear_vehicle_closing": "rear_vehicle_falling_behind",
    "rear_vehicle_falling_behind": "rear_vehicle_closing",
}


def signed_closing_speed(vehicle):
    """Read either legacy signed speed or direction-plus-magnitude input."""
    if "closing_speed_meters_per_second" in vehicle:
        return float(vehicle.get("closing_speed_meters_per_second", 0.0))
    motion = vehicle.get("gap_motion") or {}
    magnitude = float(motion.get("speed_meters_per_second", 0.0))
    if motion.get("trend") == "closing":
        return magnitude
    if motion.get("trend") == "opening":
        return -magnitude
    return 0.0


def oracle_traffic_bases(observation, closing_speed_deadband=0.1):
    """Return the two factual front/rear labels implied by an observation."""
    traffic = observation.get("traffic", {})
    facts = {}
    for subject in ("front", "rear"):
        vehicle = traffic.get(subject)
        if vehicle is None:
            label = f"no_{subject}_vehicle"
            evidence = f"traffic.{subject}=null"
        else:
            speed = signed_closing_speed(vehicle)
            if abs(speed) <= closing_speed_deadband:
                label = f"{subject}_gap_approximately_constant"
            elif subject == "front":
                label = (
                    "closing_on_front_vehicle"
                    if speed > 0.0
                    else "front_vehicle_pulling_away"
                )
            else:
                label = (
                    "rear_vehicle_closing"
                    if speed > 0.0
                    else "rear_vehicle_falling_behind"
                )
            evidence = f"{subject}.closing_speed={speed:.2f} m/s"
        facts[subject] = {"label": label, "evidence": evidence}
    return facts


def verify_traffic_basis(label, observation, closing_speed_deadband=0.1):
    """Return supported/ambiguous/contradicted using the shared oracle logic."""
    if label not in TRAFFIC_BASIS_LABELS:
        return "unknown", "unrecognized traffic basis"

    subject = "front" if "front" in label else "rear"
    fact = oracle_traffic_bases(observation, closing_speed_deadband)[subject]
    if label == fact["label"]:
        return "supported", fact["evidence"]

    vehicle = observation.get("traffic", {}).get(subject)
    moving_claims = {
        "closing_on_front_vehicle",
        "front_vehicle_pulling_away",
        "rear_vehicle_closing",
        "rear_vehicle_falling_behind",
    }
    if vehicle is not None and label in moving_claims:
        if abs(signed_closing_speed(vehicle)) <= closing_speed_deadband:
            return "ambiguous", fact["evidence"]
    return "contradicted", fact["evidence"]
