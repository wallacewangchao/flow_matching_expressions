import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os

data_from_folder = r"C:\Users\Public\ChaosProjects\flow_matching\data\unity_data"
data_output_folder = r"C:\Users\Public\ChaosProjects\flow_matching\data\unity_world_pose_data"

os.makedirs(data_output_folder, exist_ok=True)

for filename in os.listdir(data_from_folder):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(data_from_folder, filename)
        output_path = os.path.join(data_output_folder, filename)
        dict_list = []
        with open(input_path, "r") as f_in:
            for json_line in f_in:
                try:
                    data_dict = json.loads(json_line.strip())
                    # The "Head" data here is in the world coordinate system
                    # The "Eye", "LeftHand", "RightHand", and "TargetObject" data here are in the local coordinate system relative to "Head"
                    # parent_pos = [v for k, v in data_dict["Head"]["position"].items()]  # x, y, z
                    parent_pos = np.array([0,0,0])  # The "Head" position here is the origin of the world coordinate system
                    parent_rot_quat = [v for k, v in data_dict["Head"]["rotation"].items()]  # x, y, z, w
                    parent_pos = np.array(parent_pos)
                    parent_rot_quat = np.array(parent_rot_quat)  # xyzw

                    for key in ["Eye", "LeftHand", "RightHand", "TargetObject"]:
                        if key in data_dict:
                            data = data_dict[key]
                            child_local_pos = [v for k, v in data["position"].items()]  # x, y, z
                            child_local_rot_quat = [v for k, v in data["rotation"].items()]
                            child_local_pos = np.array(child_local_pos)
                            child_local_rot_quat = np.array(child_local_rot_quat)  # xyzw
                            # Note: In scipy, Rotation uses quaternion order (x, y, z, w) by default
                            parent_rot = R.from_quat(parent_rot_quat)
                            child_local_rot = R.from_quat(child_local_rot_quat)
                            # Compute the child's world position
                            child_world_pos = parent_pos + parent_rot.apply(child_local_pos)
                            # Compute the child's world rotation
                            child_world_rot = parent_rot * child_local_rot
                            child_world_quat = child_world_rot.as_quat()
                            # Convert the result to a dictionary
                            child_world_pos_dict = {
                                "x": float(child_world_pos[0]),
                                "y": float(child_world_pos[1]),
                                "z": float(child_world_pos[2])
                            }
                            child_world_rot_dict = {
                                "x": float(child_world_quat[0]),
                                "y": float(child_world_quat[1]),
                                "z": float(child_world_quat[2]),
                                "w": float(child_world_quat[3])
                            }
                            # Update the dictionary
                            data_dict[key]["position"] = child_world_pos_dict
                            data_dict[key]["rotation"] = child_world_rot_dict
                    dict_list.append(data_dict)
                except Exception as e:
                    print(f"Error processing line in {filename}: {e}")
        # Write to the new jsonl file
        with open(output_path, "w") as f_out:
            for item in dict_list:
                f_out.write(json.dumps(item) + "\n")
