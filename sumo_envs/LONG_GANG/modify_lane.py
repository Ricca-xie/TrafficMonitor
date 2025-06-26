# Re-import after code execution state reset
import xml.etree.ElementTree as ET
from xml.dom import minidom

# File paths
input_path = "C:\TrafficMonitor\TrafficMonitor\sumo_envs\LONG_GANG\env\osm.rou.xml"
output_path = "C:\TrafficMonitor\TrafficMonitor\sumo_envs\LONG_GANG\env\osm.rou.xml"

# Load XML
tree = ET.parse(input_path)
root = tree.getroot()

# Filter out all ego vehicles
vehicles = [v for v in root.findall("vehicle") if v.attrib.get("type") == "ego"]
for v in vehicles:
    root.remove(v)

# Parameters for new vehicle generation
start_time = 5
batch_interval = 180
vehicles_per_batch = 10
vehicle_interval = 2
even_lane_cycle = [0, 1, 2, 3, 0]
odd_lane_cycle = [0, 1, 0, 1, 0]
even_route = "1125684496#0 1125684496#1"
odd_route = "1131230259#0 239700065#3 239700065#4 1125678574"

# Construct new vehicles
current_time = start_time
batch_index = 0

while current_time < 2500:

    if batch_index % 2 == 0:
        selected_route = even_route
        lane_cycle = even_lane_cycle
    else:
        selected_route =odd_route
        lane_cycle = odd_lane_cycle

    for i in range(vehicles_per_batch):
        v_id = f"1125684496#0__{batch_index}__ego.{i}"
        depart_time = current_time + i * vehicle_interval
        lane_id = str(lane_cycle[i % len(lane_cycle)])

        veh = ET.Element("vehicle", {
            "id": v_id,
            "type": "ego",
            "depart": str(depart_time),
            "departLane": lane_id
        })
        ET.SubElement(veh, "route", {
            "edges": selected_route
        })
        root.append(veh)
    current_time += batch_interval
    batch_index += 1

# Pretty print and save
pretty_xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(pretty_xml)
