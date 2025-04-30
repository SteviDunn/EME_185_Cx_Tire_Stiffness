from pathlib import Path 
import yaml

import pandas as pd 
import matplotlib.pyplot as plt 

from rosbags.highlevel import AnyReader
from rosbags.typesys.stores import Stores
from rosbags.typesys import get_typestore, get_types_from_msg

print("Imports OK")

#load the metadata.yaml
meta_path = Path("Cx_sample_data/metadata.yaml")
with open(meta_path) as f:
    meta  = yaml.safe_load(f)

#list the topics in the data file
topics = [t["topic_metadata"]["name"]
    for t in meta["rosbag2_bagfile_information"]["topics_with_message_count"]]
print("Found topics:", topics)

#test for imu message 

vesc_imu_msg_text = Path("vesc_msgs/msg/VescImu.msg").read_text()
vesc_imu_stamped_msg_text = Path("vesc_msgs/msg/VescImuStamped.msg").read_text()

add_types = {}
add_types.update(get_types_from_msg(vesc_imu_msg_text, 'vesc_msgs/msg/VescImu'))
add_types.update(get_types_from_msg(vesc_imu_stamped_msg_text, 'vesc_msgs/msg/VescImuStamped'))

typestore = get_typestore(
    Stores.ROS2_FOXY,
)
typestore.register(add_types)

bag_dir = Path("Cx_sample_data")

with AnyReader([bag_dir], default_typestore= typestore) as reader:
    #grab the first imu message we see
    for conn, ts, raw in reader.messages():
        if conn.topic == "/sensors/imu":
            msg= reader.deserialize(raw, conn.msgtype)
            print("First IMU accl_x=", msg.imu.linear_acceleration.x)
            break
    #count all imu messages
    imu_conns = [c for c in reader.connections if c.topic == "/sensors/imu"]
    imu_msgs = list(reader.messages(connections=imu_conns))
    print(f"Found {len(imu_msgs)} total IMU messages")

    #extract all imu data into a list
    imu_data =[]
    for conn, ts, raw in reader.messages(connections=imu_conns):
        msg= reader.deserialize(raw, conn.msgtype)
        t= ts* 1e-9
        imu_data.append((
            t,
            msg.imu.linear_acceleration.x,
            msg.imu.linear_acceleration.y,
            msg.imu.linear_acceleration.z
            
        ))
    #print(imu_data[:30])

    #verify the motor speed tpoic
    motor_conns = [c for c in reader.connections if c.topic=="/commands/motor/speed"]
    for conn, ts, raw in reader.messages(connections=motor_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        print("Motor speed first value=", msg.data, "at",ts*1e-9, "s" )
        break
    
    #verify the count of motor messages
    motor_msgs = list(reader.messages(connections=motor_conns))
    print(f"Found {len(motor_msgs)} total motor-speed messages")

    #extract all the motor data
    start_time = None
    motor_data = []
    for conn, ts, raw in reader.messages(connections=motor_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        motor_data.append((ts*1e-9, msg.data))
    print(motor_data[500:600])

    #dump data into data frames
    imu_df   = pd.DataFrame(imu_data,   columns=["time","ax","ay","az"])
    motor_df = pd.DataFrame(motor_data, columns=["time","wheel_speed"])

    print("IMU data frame shape:", imu_df.shape)
    print(imu_df.head())

    print("Motor speed data frame shape:", motor_df.shape)
    print(motor_df.head())
