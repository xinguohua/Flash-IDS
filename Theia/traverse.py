import json
def traverse(ids, mapping, edges, hops, visited=None):
    if hops == 0:
        return set()

    if visited is None:
        visited = set()

    neighbors = set()
    for src, dst in zip(edges[0], edges[1]):
        src_mapped, dst_mapped = mapping[src], mapping[dst]

        if (src_mapped in ids and dst_mapped not in visited) or \
           (dst_mapped in ids and src_mapped not in visited):
            neighbors.add(src_mapped)
            neighbors.add(dst_mapped)

        visited.add(src_mapped)
        visited.add(dst_mapped)

    neighbors.difference_update(ids)
    return ids.union(traverse(neighbors, mapping, edges, hops - 1, visited))

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_connected_alerts(start_alert, mapping, edges, depth, remaining_alerts):
    connected_path = traverse({start_alert}, mapping, edges, depth)
    return connected_path.intersection(remaining_alerts)

def generate_incident_graphs(alerts, edges, mapping, depth):
    incident_graphs = []
    remaining_alerts = set(alerts)

    while remaining_alerts:
        alert = remaining_alerts.pop()
        connected_alerts = find_connected_alerts(alert, mapping, edges, depth, remaining_alerts)

        if len(connected_alerts) > 1:
            incident_graphs.append(connected_alerts)
            remaining_alerts -= connected_alerts

    return incident_graphs