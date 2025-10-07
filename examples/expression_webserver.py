import time
import sys
import json
import signal
import threading
import collections
import queue
import socket

import keyboard
from websocket_server import WebsocketServer
from save_data import SaveData
from termcolor import colored
import os
from flow_expression_inference_class import FlowExpression
import numpy as np
from copy import deepcopy


class WebSocketServerApp:
    def __init__(
        self,
        host="0.0.0.0",
        port=20000,
        is_inference_mode=False,
    ):
        self.host = host
        self.port = port
        self.save_data_key = "t"
        self.send_fm_list_data_key = "o"
        self.send_dummy_dict_data_key = "p"
        self.data_saver = SaveData()
        self.is_receiving = False
        self.is_inference_mode = is_inference_mode
        self.is_parallel_sending = True
        self.send_sleep_time = 0.1

        print(f"is inference mode : {self.is_inference_mode}")

        self.running = True
        self.key_detection_thread = None
        self.pack_data_to_npz_thread = None
        self.read_and_send_thread = None
        self.data_to_send_folder = "data/dummy_data_to_send"

        # Set up the websocket server
        self.ws_server = WebsocketServer(host=self.host, port=self.port)

        self.is_sent_enough_data = threading.Event()
        self.is_request_prempt = threading.Event()
        self.data_to_send_queue = queue.Queue()
        self.data_to_send_thread = threading.Thread(
            target=self.send_data_worker, daemon=True
        )
        self.data_to_send_thread.start()

        if is_inference_mode:
            self.fm_expression = FlowExpression()
            print("FlowExpression Model Loaded")

            self.OBS_PARMS = {
            "obs_horizon": self.fm_expression.obs_horizon,
            "pred_horizon": self.fm_expression.pred_horizon,
            "action_horizon": self.fm_expression.pred_horizon - 1,
            "obs_poses_dim": self.fm_expression.obs_poses_dim,
            }

            self.fm_inference_worker_thread = threading.Thread(
                target=self.FM_inference_worker, daemon=True
            )
            self.inference_in_progress = threading.Event()
            self.fm_worker_queue = queue.Queue()
            self.real_time_obs_queue = collections.deque(
                maxlen=self.OBS_PARMS["obs_horizon"]
            )
            self.obs_queue_lock = threading.Lock()  # add a lock

            self.fm_inference_worker_thread.start()

        self.dummy_obs_input_dict = {}
        self._read_jsonl_to_FM_obs_data(self.OBS_PARMS["obs_horizon"])

    def signal_handler(self, sig, frame):
        print("Ctrl+C pressed, shutting down...")
        self.running = False
        if self.key_detection_thread and self.key_detection_thread.is_alive():
            self.key_detection_thread.join(timeout=1)
        if self.is_inference_mode and self.fm_inference_worker_thread and self.fm_inference_worker_thread.is_alive():
            self.fm_inference_worker_thread.join(timeout=1)
        sys.exit(0)

    def new_client(self, client, server):
        print(colored("client connected: ", "green"), client["id"])
        response = "Server connected!"
        server.send_message(client, response)

    def client_left(self, client, server):
        print(colored("client disconnected", "red"), client["id"])

    def message_received(self, client, server, message):
        # print(colored("message received: ", "yellow"), message)
        print(colored(f"received from Unity: {len(message)} bytes","green"))
        if message == "start_transmit":
            print(colored("start_transmit", "green"))
            self.is_receiving = True

        elif message == "stop_transmit":
            print(colored("stop_transmit", "red"))
            if self.is_inference_mode:
                with self.obs_queue_lock:
                    self.real_time_obs_queue.clear()
            else:
                self.data_saver.save_as_json()
                self.data_saver.renew_dats()
            self.is_receiving = False

        elif message == "Hello from Unity!":
            print(colored("Unity Connected", "green"))

        else:    
            try:
                data = json.loads(message)
                if self.is_inference_mode:
                    # If in inference mode, start real-time inference.
                    with self.obs_queue_lock:
                        # Add new observation to the queue:
                        # only keep the latest obs_horizon observations,
                        # by using deque with maxlen
                        self.real_time_obs_queue.append(data)
                        
                        # Put data into the fm_worker_queue if:
                        # 1) there are enough observations
                        if len(self.real_time_obs_queue) == self.OBS_PARMS["obs_horizon"]:
                            # 2) not already in the middle of an inference
                            if not self.inference_in_progress.is_set():
                                self.fm_worker_queue.put(
                                    list(deepcopy(self.real_time_obs_queue))
                                )
                                print("data added to inference queue")
                else:
                    # If in collect mode, save the data
                    self.data_saver.update(data)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")

    def send_data(self, dict_list: list, time_interval: float):
        """
        Serializes the provided dictionary to JSON and sends it to all connected websocket clients.
        """
        try:
            for data in dict_list:

                json_data = json.dumps(data)
                self.ws_server.send_message_to_all(json_data)
                print(f"Json data sent to Unity: {len(json_data)} bytes")
                time.sleep(time_interval)

        except Exception as e:
            print(f"Error sending data: {e}")

    def send_data_worker(self):
        while self.running:
            dict_list = self.data_to_send_queue.get(block=True)
            self.is_sent_enough_data.clear()
            self.is_request_prempt.clear()

            for i in range(len(dict_list)):
                if self.is_request_prempt.is_set():
                    print(colored("preemption request", "red"))
                    break
                s = json.dumps(dict_list[i])
                self.ws_server.send_message_to_all(s)
                if i == self.OBS_PARMS["action_horizon"] - 1:
                    self.is_sent_enough_data.set()
                print(f"Json data sent to Unity: {len(s)} bytes")
                time.sleep(self.send_sleep_time)

    def read_and_send_stored_data(self):
        dict_list = []
        for filename in os.listdir(self.data_to_send_folder):
            if filename.endswith(".jsonl"):
                with open(os.path.join(self.data_to_send_folder, filename), "r") as f:
                    for line in f:
                        dict_list.append(json.loads(line.strip()))
        print(
            f"""read {len(dict_list)} data points, sending in every {self.send_sleep_time} second."""
        )
        self.send_data(dict_list=dict_list, time_interval=self.send_sleep_time)

    def FM_inference_stored_data_and_send(self, obs_input_dict: dict):
        with open(
            r"C:\Users\Public\ChaosProjects\flow_matching_chao_local\data\unity_world_pose_data\unity_data_20250421_175241.jsonl",
            "r",
        ) as f:
            l = []
            for line in f:
                l.append(
                    json.loads(line.strip())
                )  # take each line as a action point in the episode
            # l is the list of all action points in the episode

        # Loop through each observation horizon in the data list
        # and put them into the fm_worker_queue one by one
        self.is_sent_enough_data.set()
        for i in range(len(l) - self.OBS_PARMS["obs_horizon"] + 1):
            self.is_sent_enough_data.wait()
            self.fm_worker_queue.put(
                l[i : i + self.OBS_PARMS["obs_horizon"]]
            )
            print("data added to inference queue")

    def FM_inference_worker(self):
        while self.running:
            obs = self.fm_worker_queue.get(block=True)
            self.inference_in_progress.set()

            obs_dict = self._convert_dict_to_FM_obs_data(
                obs_list=obs,
                obs_horizon=self.OBS_PARMS["obs_horizon"],
            )

            action_pred_list = self.fm_expression.inference(obs_dict)
            a_dict_list = self._convert_FM_data_to_dict(action_pred_list)

            if self.is_parallel_sending:
                self.is_request_prempt.set()
                self.data_to_send_queue.put(a_dict_list)

                self.is_sent_enough_data.wait()
            else:
                self.send_data(dict_list=a_dict_list, time_interval=0.1)

            self.inference_in_progress.clear()

    def _convert_dict_to_FM_obs_data(
        self, obs_list: list, obs_horizon: int
    ):
        """
        Convert the data from Unity to np format required by Flow Matching
        """

        (obs_poses, obs_lang, _, _) = self.data_saver.flatten_data(d_list=obs_list)
        poses_horizon = [
            obs_poses[i : i + obs_horizon]
            for i in range(0, len(obs_poses) - obs_horizon + 1)
        ]
        lang_horizon = [
            obs_lang[i : i + obs_horizon]
            for i in range(0, len(obs_lang) - obs_horizon + 1)
        ]
        return {
            "obs_object_poses": np.array(poses_horizon),
            "obs_languages": np.array(lang_horizon),
        }

    def _read_jsonl_to_FM_obs_data(
        self, obs_horizon: int
    ) -> list:

        with open(
            r"C:\Users\Public\ChaosProjects\flow_matching_chao_local\data\unity_world_pose_data\unity_data_20250421_175241.jsonl",
            "r",
        ) as f:
            l = []
            for line in f:
                l.append(
                    json.loads(line.strip())
                )  # take each line as a action point in the episode
            # l is the list of all action points in the episode

            (obs_poses, obs_lang, _, _) = self.data_saver.flatten_data(d_list=l)
            poses_horizon = [
                obs_poses[i : i + obs_horizon]
                for i in range(0, len(obs_poses) - obs_horizon + 1)
            ]
            lang_horizon = [
                obs_lang[i : i + obs_horizon]
                for i in range(0, len(obs_lang) - obs_horizon + 1)
            ]
            self.dummy_obs_input_dict["obs_object_poses"] = np.array(poses_horizon)
            self.dummy_obs_input_dict["obs_languages"] = np.array(lang_horizon)

    def _convert_FM_data_to_dict(self, fm_list: list) -> list:
        action_dict_list = []
        for l in fm_list:
            action_dict = {}

            """
            Action Index Mapping:

            Head: 0-4 (rotation only)
            Eye: 4-8 (rotation only)
            LeftHand: 8-15 (position + rotation)
            RightHand: 15-22 (position + rotation)
            FacialExpression: 22-30 (blend shapes)

            Total: 30
            """

            action_dict["Head"] = self._pose_list_to_dict(l[0:4], with_position=False)
            action_dict["Eye"] = self._pose_list_to_dict(l[4:8], with_position=False)

            action_dict["LeftHand"] = self._pose_list_to_dict(
                l[8:15], with_position=True
            )
            action_dict["RightHand"] = self._pose_list_to_dict(
                l[15:22], with_position=True
            )

            action_dict["FacialExpression"] = {}
            for i, name in enumerate(self.data_saver.facial_expression_names):
                action_dict["FacialExpression"][name] = l[22 + i]

            action_dict_list.append(action_dict)

        return action_dict_list

    def _pose_list_to_dict(
        self,
        pose_list: list,
        with_position: bool = True,
    ):
        """
        Convert a list of pose to a dictionary
        """
        pose_dict = {"rotation": {}}
        if with_position:
            pose_dict["position"] = {}

            pose_dict["position"]["x"] = pose_list[0]
            pose_dict["position"]["y"] = pose_list[1]
            pose_dict["position"]["z"] = pose_list[2]
            pose_dict["rotation"]["x"] = pose_list[3]
            pose_dict["rotation"]["y"] = pose_list[4]
            pose_dict["rotation"]["z"] = pose_list[5]
            pose_dict["rotation"]["w"] = pose_list[6]
        else:
            pose_dict["rotation"]["x"] = pose_list[0]
            pose_dict["rotation"]["y"] = pose_list[1]
            pose_dict["rotation"]["z"] = pose_list[2]
            pose_dict["rotation"]["w"] = pose_list[3]

        return pose_dict

    def start_key_detector(self):
        keyboard.on_release_key(self.save_data_key, lambda _: self._handle_save_key())
        keyboard.on_release_key("i", lambda _: self._handle_i_key())
        keyboard.on_release_key("p", lambda _: self._handle_p_key())
        keyboard.wait()

    def _handle_save_key(self):
        if self.is_inference_mode:
            print("In inference mode, not saving data.")
            return
        print("t key pressed")
        self.data_saver.save_json_into_npz()

    def _handle_i_key(self):
        print("i key pressed")
        self.read_and_send_stored_data()

    def _handle_p_key(self):
        print("p key pressed")
        self.FM_inference_stored_data_and_send(self.dummy_obs_input_dict)


    def _get_server_ip(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return self.host


    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        print("WS Server Established")
        server_ip = self._get_server_ip()
        print(
            colored(
                f"Server IP: ws://{server_ip}:{self.port} (binding: {self.host})",
                "cyan",
            )
        )
        print(
            colored(
                f'Press "{self.save_data_key}" to pack json data into .npy.', "green"
            )
        )

        self.ws_server.set_fn_new_client(self.new_client)
        self.ws_server.set_fn_client_left(self.client_left)
        self.ws_server.set_fn_message_received(self.message_received)

        self.key_detection_thread = threading.Thread(
            target=self.start_key_detector, daemon=True
        )
        self.key_detection_thread.start()
        self.ws_server.run_forever()


if __name__ == "__main__":

        # Check if an argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'collect' or 'inference'")
        sys.exit(1)

    arg = sys.argv[1].lower()
    if arg == "collect":
        is_inference_mode = False
    elif arg == "inference":
        is_inference_mode = True
    else:
        print("Invalid argument. Please specify 'collect' or 'inference'")
        sys.exit(1)

    app = WebSocketServerApp(is_inference_mode=is_inference_mode)
    app.run()
