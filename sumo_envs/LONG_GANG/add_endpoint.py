# Re-import required packages after code state reset
import xml.etree.ElementTree as ET

# 重新设定输入输出路径
input_path = "C:\TrafficMonitor\TrafficMonitor\sumo_envs\LONG_GANG\env\osm.rou.xml"
output_path = "C:\TrafficMonitor\TrafficMonitor\sumo_envs\LONG_GANG\env\osm.rou.xml"

# 读取原始文件
tree = ET.parse(input_path)
root = tree.getroot()

# 获取现有ego车辆的最后depart时间
existing_depart_times = []
for veh in root.findall("vehicle"):
    if veh.attrib.get("type") == "ego":
        depart_time = float(veh.attrib["depart"])
        existing_depart_times.append(depart_time)
max_depart_time = int(max(existing_depart_times)) if existing_depart_times else 0

# 持续时间设定
end_time = 2500
interval = 120
vehicles_per_batch = 5

# 计算下一批编号起点
last_batch_index = len(existing_depart_times) // vehicles_per_batch

# 生成新的车辆条目直到2500秒
next_depart_time = max_depart_time + 1
batch_index = last_batch_index + 1
while next_depart_time < end_time:
    for i in range(vehicles_per_batch):
        veh = ET.Element("vehicle", {
            "id": f"1125684496#0__{batch_index}__ego.{i}",
            "type": "ego",
            "depart": str(next_depart_time + i),
            "departLane": str(i % 4)
        })
        route = ET.SubElement(veh, "route", {
            "edges": "1125684496#0 1125684496#1"
        })
        root.append(veh)
    next_depart_time += interval
    batch_index += 1

tree.write(output_path, encoding="utf-8", xml_declaration=True)
