from pathlib import Path
import yaml

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from rosbags.highlevel import AnyReader
from rosbags.typesys.stores import Stores
from rosbags.typesys import get_typestore, get_types_from_msg

print("Imports OK")

# Constants
grav_mass = 3331.5  # grams
mass_kg = grav_mass / 1000.0  # convert to kg
R_eff = 0.30  # effective rolling radius (m)

# Gear ratios
gearbox_ratio = 54 / 13
diff_ratio    = 37 / 13
gear_total   = gearbox_ratio * diff_ratio

# Paths
bag_dir   = Path("Cx_sample_data/EME_185_Run_2")
meta_path = bag_dir / "metadata.yaml"

# Load metadata
with open(meta_path) as f:
    meta = yaml.safe_load(f)
topics = [t["topic_metadata"]["name"]
          for t in meta["rosbag2_bagfile_information"]["topics_with_message_count"]]
print("Found topics:", topics)

# Register custom VESC message types
vesc_imu_txt        = Path("vesc_msgs/msg/VescImu.msg").read_text()
vesc_imu_stamped_txt = Path("vesc_msgs/msg/VescImuStamped.msg").read_text()
add_types = {}
add_types.update(get_types_from_msg(vesc_imu_txt, 'vesc_msgs/msg/VescImu'))
add_types.update(get_types_from_msg(vesc_imu_stamped_txt, 'vesc_msgs/msg/VescImuStamped'))

typestore = get_typestore(Stores.ROS2_FOXY)
typestore.register(add_types)

# Read bag and unpack
with AnyReader([bag_dir], default_typestore=typestore) as reader:
    # Extract sensor streams into DataFrames
    imu_conns   = [c for c in reader.connections if c.topic == "/sensors/imu"]
    motor_conns = [c for c in reader.connections if c.topic == "/commands/motor/speed"]
    odom_conns  = [c for c in reader.connections if c.topic == "/odom"]
    lidar_conns = [c for c in reader.connections if c.topic == "/LIDAR_velocity"]

    # IMU data: time, ax, ay, az
    imu_data = []
    for conn, ts, raw in reader.messages(connections=imu_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        t   = ts * 1e-9
        imu_data.append((t,
                         msg.imu.linear_acceleration.x,
                         msg.imu.linear_acceleration.y,
                         msg.imu.linear_acceleration.z))
    imu_df = pd.DataFrame(imu_data, columns=["time","ax","ay","az"]).sort_values("time")
    print("IMU data frame shape:", imu_df.shape)
    print(imu_df.head(), "\n")

    # Motor data: time, motor_rpm
    motor_data = []
    for conn, ts, raw in reader.messages(connections=motor_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        t   = ts * 1e-9
        motor_data.append((t, msg.data))
    motor_df = pd.DataFrame(motor_data, columns=["time","motor_rpm"]).sort_values("time")
    print("Motor RPM data frame shape:", motor_df.shape)
    print(motor_df.head(), "\n")

    # Convert motor RPM -> wheel_speed (rad/s)
    motor_df["omega_motor"] = motor_df["motor_rpm"] * (2 * np.pi / 60)
    motor_df["wheel_speed"] = motor_df["omega_motor"] / gear_total
    print("Converted wheel_speed head:")
    print(motor_df[["time","wheel_speed"]].head(), "\n")

    # Odometry data: time, v_odom
    odom_data = []
    for conn, ts, raw in reader.messages(connections=odom_conns):
        if conn.msgtype != "nav_msgs/msg/Odometry":
            continue
        msg = reader.deserialize(raw, conn.msgtype)
        t   = ts * 1e-9
        odom_data.append((t, msg.twist.twist.linear.x))
    odom_df = pd.DataFrame(odom_data, columns=["time","v_odom"]).sort_values("time")
    print("Odometry data frame shape:", odom_df.shape)
    print(odom_df.head(), "\n")

    # LIDAR data: time, v_lidar
    lidar_data = []
    for conn, ts, raw in reader.messages(connections=lidar_conns):
        msg = reader.deserialize(raw, conn.msgtype)
        t   = ts * 1e-9
        lidar_data.append((t, msg.twist.linear.x))
    lidar_df = pd.DataFrame(lidar_data, columns=["time","v_lidar"]).sort_values("time")
    print("LIDAR data frame shape:", lidar_df.shape)
    print(lidar_df.head(), "\n")

    # Merge DataFrames: motor+odom -> df, then add IMU and LIDAR
    df = pd.merge_asof(
        motor_df[["time","wheel_speed"]],
        odom_df,
        on="time",
        direction="nearest",
        tolerance=0.01
    )
    df = pd.merge_asof(
        df,
        imu_df[["time","ax","ay","az"]],
        on="time",
        direction="nearest",
        tolerance=0.01
    )
    if not lidar_df.empty:
        # ensure time is numeric
        lidar_df["time"] = pd.to_numeric(lidar_df["time"], errors="coerce")
        df = pd.merge_asof(
            df,
            lidar_df,
            on="time",
            direction="nearest",
            tolerance=0.01
    )
    else:
        print("⚠️  No LIDAR data found; skipping LIDAR merge.")
    print("Merged DataFrame shape:", df.shape)
    print(df.head(), "\n")

    # Compute filtered acceleration
    df["ax_filt"] = df["ax"].rolling(window=5, center=True).mean()
    bias = df["ax_filt"].loc[df["time"] < df["time"].min() + 1].mean()
    df["ax_filt"] -= bias

    # Compute force and slip ratio
    df["F_x_meas"] = mass_kg * df["ax_filt"]
    df["v_wheel"]   = df["wheel_speed"] * R_eff
    df["kappa"]     = (df["v_wheel"] - df["v_odom"]) / df["v_odom"]

    # Estimate Cx via zero-intercept regression
    from sklearn.linear_model import LinearRegression
    fit_df = df.dropna(subset=["kappa","F_x_meas"]).query("v_odom > 1e-3")
    X = fit_df["kappa"].values.reshape(-1,1)
    y = fit_df["F_x_meas"].values
    model = LinearRegression(fit_intercept=False).fit(X, y)
    Cx_est = model.coef_[0]
    print(f"Estimated Cx = {Cx_est:.2f} N per unit slip", "\n")

    # Plotting
    plt.figure()
    plt.plot(df["time"], df["v_odom"],  label="Ground speed")
    plt.plot(df["time"], df["v_wheel"], label="Wheel speed")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    df_slip = df[df["v_odom"] > 1e-3]
    plt.plot(df_slip["time"], df_slip["kappa"] )
    plt.xlabel("Time (s)")
    plt.ylabel("Slip ratio κ")
    plt.title("Slip Ratio vs Time")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(fit_df["kappa"], fit_df["F_x_meas"], s=5, alpha=0.6)
    plt.plot(fit_df["kappa"], Cx_est * fit_df["kappa"], color="red", linewidth=2)
    plt.xlabel("Slip ratio κ")
    plt.ylabel("Longitudinal Force Fₓ (N)")
    plt.title(f"Force vs Slip Ratio — Cx ≈ {Cx_est:.2f}")
    plt.tight_layout()
    plt.show()
