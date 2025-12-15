"""
Data Logger for Unitree G1 robot - No ROS2/Mocap version
Collects robot state and command data via Unitree SDK only.
Works reliably over SSH.
"""
import time
import sys
import signal
import numpy as np
from collections import deque
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
import threading
import select
import os
import argparse
import yaml

sys.path.append('./rl_policy/..')
from utils.robot import Robot


class DataLogger:
    def __init__(self, save_path='data_log.npz', config=None, log_interval=1.0, buffer_rate=50.0, buffer_size=50*600):
        """
        Initializes the DataLogger (no ROS2, no mocap).

        Args:
            save_path (str): Path to save the .npz log file.
            log_interval (float): Interval in seconds for throttled logging.
            buffer_rate (float): Frequency in Hz to update the observation buffer.
            buffer_size (int): Maximum number of entries to store in the buffer.
        """
        self.config = config
        self.save_path = save_path
        self.log_interval = log_interval
        self.buffer_rate = buffer_rate
        self.running = True
        
        # Import the correct message types based on robot
        if self.config["ROBOT_TYPE"] == "h1" or self.config["ROBOT_TYPE"] == "go2":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        elif self.config["ROBOT_TYPE"] == "g1_29dof" or self.config["ROBOT_TYPE"] == "h1-2":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        else:
            raise NotImplementedError(f"Robot type {self.config['ROBOT_TYPE']} is not supported yet")
        
        self.robot = Robot(config)
        self.num_actions = self.robot.NUM_JOINTS
        
        self.start_recording = False
        self.motion_episode_cnt = 0

        # Initialize buffer for high-frequency data
        self.obs_buffer = deque(maxlen=buffer_size)

        # Initialize latest data holders
        self.latest_low_state = None
        self.latest_low_cmd = None
        self.last_log_time = time.time()

        # Initialize subscribers for Unitree data
        self.sub_state = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_state.Init(self.LowStateHandler, 10)
                
        self.sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub_cmd.Init(self.LowCmdHandler, 10)

        # Set up the signal handler
        signal.signal(signal.SIGINT, self.sigINT_handler)

        print("[INFO] DataLogger initialized (no mocap mode).")

        # Start keyboard listener thread
        self.key_listener_thread = threading.Thread(target=self.start_key_listener, daemon=True)
        self.key_listener_thread.start()
        
        # Start buffer update thread
        self.buffer_thread = threading.Thread(target=self.buffer_update_loop, daemon=True)
        self.buffer_thread.start()
        
        # Start logging thread
        self.log_thread = threading.Thread(target=self.log_loop, daemon=True)
        self.log_thread.start()

    def log(self, msg):
        """Simple logging function."""
        print(f"[INFO] {msg}")

    def warn(self, msg):
        """Simple warning function."""
        print(f"[WARN] {msg}")

    def clear_buffer(self):
        """Clears the observation buffer."""
        self.obs_buffer.clear()
        self.log("Buffer cleared.")

    def start_key_listener(self):
        """Start a key listener using stdin (works reliably over SSH)."""
        print("\n" + "="*50)
        print("CONTROLS (type key + press Enter):")
        print("  ;  = START recording")
        print("  '  = STOP and save")
        print("  q  = Quit")
        print("="*50 + "\n")
        
        while self.running:
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.readline().strip()
                    if key == ";":
                        self.start_recording = True
                        self.clear_buffer()
                        self.log(f"Start recording for episode {self.motion_episode_cnt}")
                    elif key == "'":
                        self.start_recording = False
                        self.log(f"Stop recording for episode {self.motion_episode_cnt}")
                        self.process_and_save_data()
                        self.clear_buffer()
                        self.motion_episode_cnt += 1
                    elif key == "q":
                        self.log("Quitting...")
                        self.running = False
                        break
            except Exception:
                pass

    def LowStateHandler(self, msg):
        """Handles incoming LowState_ messages and stores the latest data."""
        self.latest_low_state = {
            'timestamp': time.time(),
            'motor_positions': np.array([motor.q for motor in msg.motor_state[:self.num_actions]]),
            'motor_velocities': np.array([motor.dq for motor in msg.motor_state[:self.num_actions]]),
            'motor_torques': np.array([motor.tau_est for motor in msg.motor_state[:self.num_actions]]),
            'imu_quaternion': np.array(msg.imu_state.quaternion),
            'imu_gyroscope': np.array(msg.imu_state.gyroscope),
            'imu_accelerometer': np.array(msg.imu_state.accelerometer)
        }

    def LowCmdHandler(self, msg):
        """Handles incoming LowCmd_ messages and stores the latest data."""
        self.latest_low_cmd = {
            'timestamp': time.time(),
            'motor_commands': {
                "q": np.array([motor_cmd.q for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "dq": np.array([motor_cmd.dq for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "kp": np.array([motor_cmd.kp for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "kd": np.array([motor_cmd.kd for motor_cmd in msg.motor_cmd[:self.num_actions]]),
                "tau": np.array([motor_cmd.tau for motor_cmd in msg.motor_cmd[:self.num_actions]])
            }
        }

    def log_loop(self):
        """Logs status at specified intervals."""
        while self.running:
            current_time = time.time()
            if (current_time - self.last_log_time) >= self.log_interval:
                buffer_length = len(self.obs_buffer)
                recording_status = "RECORDING" if self.start_recording else "IDLE"
                print(f"\r[{recording_status}] Buffer: {buffer_length} samples | Episode: {self.motion_episode_cnt}", end="", flush=True)
                self.last_log_time = current_time
            time.sleep(0.1)

    def buffer_update_loop(self):
        """Updates the observation buffer at specified rate."""
        period = 1.0 / self.buffer_rate
        while self.running:
            if self.start_recording and self.latest_low_state and self.latest_low_cmd:
                obs = {
                    'timestamp': time.time(),
                    'low_state': self.latest_low_state,
                    'low_cmd': self.latest_low_cmd,
                }
                self.obs_buffer.append(obs)
            time.sleep(period)

    def sigINT_handler(self, sig, frame):
        """Handles SIGINT (Ctrl+C) to gracefully exit."""
        print(f"\n[INFO] Exiting... Total episodes: {self.motion_episode_cnt}")
        print(f"[INFO] Data saved to {self.save_path}")
        self.running = False
        sys.exit(0)

    def process_and_save_data(self):
        """Processes the observation buffer and saves it into a .npz file."""
        if len(self.obs_buffer) == 0:
            self.warn("Buffer is empty, nothing to save!")
            return
            
        # Initialize lists for processed data
        buffer_timestamps = []
        buffer_low_state_motor_positions = []
        buffer_low_state_motor_velocities = []
        buffer_low_state_motor_torques = []
        buffer_low_state_imu_quaternion = []
        buffer_low_state_imu_gyroscope = []
        buffer_low_state_imu_accelerometer = []

        buffer_low_cmd_q = []
        buffer_low_cmd_dq = []
        buffer_low_cmd_kp = []
        buffer_low_cmd_kd = []
        buffer_low_cmd_tau = []

        # Iterate through the buffer and extract data
        for obs in self.obs_buffer:
            buffer_timestamps.append(obs['timestamp'])

            # Low State
            low_state = obs['low_state']
            buffer_low_state_motor_positions.append(low_state['motor_positions'])
            buffer_low_state_motor_velocities.append(low_state['motor_velocities'])
            buffer_low_state_motor_torques.append(low_state['motor_torques'])
            buffer_low_state_imu_quaternion.append(low_state['imu_quaternion'])
            buffer_low_state_imu_gyroscope.append(low_state['imu_gyroscope'])
            buffer_low_state_imu_accelerometer.append(low_state['imu_accelerometer'])

            # Low Command
            low_cmd = obs['low_cmd']
            buffer_low_cmd_q.append(low_cmd['motor_commands']['q'])
            buffer_low_cmd_dq.append(low_cmd['motor_commands']['dq'])
            buffer_low_cmd_kp.append(low_cmd['motor_commands']['kp'])
            buffer_low_cmd_kd.append(low_cmd['motor_commands']['kd'])
            buffer_low_cmd_tau.append(low_cmd['motor_commands']['tau'])

        # Convert lists to NumPy arrays
        buffer_timestamps = np.array(buffer_timestamps)
        buffer_low_state_motor_positions = np.array(buffer_low_state_motor_positions)
        buffer_low_state_motor_velocities = np.array(buffer_low_state_motor_velocities)
        buffer_low_state_motor_torques = np.array(buffer_low_state_motor_torques)
        buffer_low_state_imu_quaternion = np.array(buffer_low_state_imu_quaternion)
        buffer_low_state_imu_gyroscope = np.array(buffer_low_state_imu_gyroscope)
        buffer_low_state_imu_accelerometer = np.array(buffer_low_state_imu_accelerometer)

        buffer_low_cmd_q = np.array(buffer_low_cmd_q)
        buffer_low_cmd_dq = np.array(buffer_low_cmd_dq)
        buffer_low_cmd_kp = np.array(buffer_low_cmd_kp)
        buffer_low_cmd_kd = np.array(buffer_low_cmd_kd)
        buffer_low_cmd_tau = np.array(buffer_low_cmd_tau)

        # Ensure save directory exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            
        # Save data into npz file
        npz_file = os.path.join(self.save_path, f"motion_{self.motion_episode_cnt}.npz")
        
        print(f"\n\n[INFO] Saving {len(self.obs_buffer)} samples to {npz_file}")
        np.savez(
            npz_file, 
            time=buffer_timestamps, 
            joint_pos=buffer_low_state_motor_positions, 
            joint_vel=buffer_low_state_motor_velocities, 
            tau_est=buffer_low_state_motor_torques,
            IMU_quaternion=buffer_low_state_imu_quaternion, 
            IMU_gyro=buffer_low_state_imu_gyroscope, 
            IMU_acc=buffer_low_state_imu_accelerometer,
            joint_pos_cmd=buffer_low_cmd_q,
            joint_vel_cmd=buffer_low_cmd_dq,
            kp=buffer_low_cmd_kp,
            kd=buffer_low_cmd_kd,
            tau_cmd=buffer_low_cmd_tau,
        )

        self.log(f"Data saved to {npz_file}")
        
        # Print summary
        data = np.load(npz_file, allow_pickle=True)
        print("Saved keys:", list(data.files))
        print(f"Samples: {len(buffer_timestamps)}")

    def run(self):
        """Main run loop."""
        self.log("Data logger running. Waiting for commands...")
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.sigINT_handler(None, None)


def main():
    parser = argparse.ArgumentParser(description='Data Logger for Unitree Robot (No Mocap)')
    parser.add_argument('--config', type=str, default='config/h1.yaml', help='config file')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    args = parser.parse_args()
    
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    exp_name = args.exp_name

    save_path = './../humanoidverse/logs/delta_a_realdata'
    save_path = os.path.join(save_path, exp_name)
    current_timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_path, current_timestamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Initialize Unitree SDK
    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])

    # Create and run the data logger
    data_logger = DataLogger(save_path=save_path, config=config)
    data_logger.run()


if __name__ == '__main__':
    main()
