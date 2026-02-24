from typing import Dict, List, Tuple

# Priority order matters: if a column matches multiple groups,
# it gets assigned to the first group it matches.
GROUP_RULES: List[Tuple[str, List[str]]] = [
    ("engine", ["RPM", "EGT", "Fuel", "Torque", "Oil", "CHT", "Manifold", "Temp"]),
    ("controls", ["Elevator", "Aileron", "Rudder", "Trim", "Flap", "Throttle", "Servo", "AP ", "Autopilot"]),
    ("navigation", ["GPS", "Fix", "Satellite", "Track", "Latitude", "Longitude"]),
    ("flight_dynamics", ["Airspeed", "Ground Speed", "Altitude", "Vertical", "Pitch", "Roll", "Yaw", "Heading", "Acceleration", "Load"]),
]

def assign_groups(columns: List[str]) -> Dict[str, List[str]]:
    groups = {name: [] for name, _ in GROUP_RULES}
    groups["other"] = []

    for col in columns:
        assigned = False
        for name, keys in GROUP_RULES:
            if any(k.lower() in col.lower() for k in keys):
                groups[name].append(col)
                assigned = True
                break
        if not assigned:
            groups["other"].append(col)

    return groups


def summarize(groups: Dict[str, List[str]]):
    for k, v in groups.items():
        print(f"{k:16s} -> {len(v)}")
