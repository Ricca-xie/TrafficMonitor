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
batch_interval = 160
vehicles_per_batch = 10
vehicle_interval = 2
even_lane_cycle = [0, 1, 2, 3, 0]
odd_lane_cycle = [0, 1, 0, 1, 0]
even_id = "1125684496"
odd_id = "1131230259"
even_route = "1125684496#0 1125684496#1"
odd_route = "1131230259#0 239700065#3 239700065#4 1125678574"

# Construct new vehicles
current_time = start_time
batch_index = 0

while current_time < 800:
    # Generate even and odd lanes simultaneously for first batch
    if batch_index == 0:
        # Generate even lane vehicles (first group)
        for i in range(vehicles_per_batch):
            v_id = f"{even_id}#0__{batch_index}__ego.{i}"
            depart_time = current_time + i * vehicle_interval
            lane_id = str(even_lane_cycle[i % len(even_lane_cycle)])

            veh = ET.Element("vehicle", {
                "id": v_id,
                "type": "ego",
                "depart": str(depart_time),
                "departLane": lane_id
            })
            ET.SubElement(veh, "route", {"edges": even_route})
            root.append(veh)

        # Generate odd lane vehicles (second group)
        for i in range(vehicles_per_batch):
            v_id = f"{odd_id}#0__{batch_index}__ego.{i}"
            depart_time = current_time + i * vehicle_interval
            lane_id = str(odd_lane_cycle[i % len(odd_lane_cycle)])

            veh = ET.Element("vehicle", {
                "id": v_id,
                "type": "ego",
                "depart": str(depart_time),
                "departLane": lane_id
            })
            ET.SubElement(veh, "route", {"edges": odd_route})
            root.append(veh)

    # Generate third group (odd lane only) 10 seconds after first batch ends
    elif batch_index == 1:
        third_group_start = current_time + 10  # 10 seconds after first batch

        for i in range(vehicles_per_batch):
            v_id = f"{odd_id}#0__{batch_index}__ego.{i}"
            depart_time = third_group_start + i * vehicle_interval
            lane_id = str(odd_lane_cycle[i % len(odd_lane_cycle)])

            veh = ET.Element("vehicle", {
                "id": v_id,
                "type": "ego",
                "depart": str(depart_time),
                "departLane": lane_id
            })
            ET.SubElement(veh, "route", {"edges": odd_route})
            root.append(veh)

    # Subsequent batches follow normal interval
    else:
        # Generate odd lane vehicles
        for i in range(vehicles_per_batch):
            v_id = f"{odd_id}#0__{batch_index}__ego.{i}"
            depart_time = current_time + i * vehicle_interval
            lane_id = str(odd_lane_cycle[i % len(odd_lane_cycle)])

            veh = ET.Element("vehicle", {
                "id": v_id,
                "type": "ego",
                "depart": str(depart_time),
                "departLane": lane_id
            })
            ET.SubElement(veh, "route", {"edges": odd_route})
            root.append(veh)

    current_time += batch_interval
    batch_index += 1

# Pretty print and save
pretty_xml = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(pretty_xml)