import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog, QDialog
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import psutil
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Data Collection Function
def collect_data():
    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception as e:
        cpu_freq = None
    data = {
        'timestamp': datetime.now(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_read_bytes': psutil.disk_io_counters().read_bytes,
        'disk_write_bytes': psutil.disk_io_counters().write_bytes,
        'network_sent_bytes': psutil.net_io_counters().bytes_sent,
        'network_recv_bytes': psutil.net_io_counters().bytes_recv,
        'cpu_freq': cpu_freq,
        'available_memory': psutil.virtual_memory().available,
        'swap_memory_percent': psutil.swap_memory().percent,
        'disk_read_time': psutil.disk_io_counters().read_time,
        'disk_write_time': psutil.disk_io_counters().write_time,
        'network_packets_sent': psutil.net_io_counters().packets_sent,
        'network_packets_recv': psutil.net_io_counters().packets_recv
    }
    return data

# Custom Dataset for LSTM
class SystemDataDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+1], dtype=torch.float32), torch.tensor(self.data[idx+1:idx+2], dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
    
    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# RL Environment
class EnhancedComplexSystemEnv(gym.Env):
    def __init__(self, data_df):
        super(EnhancedComplexSystemEnv, self).__init__()
        self.data_df = data_df
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32)
        self.max_steps = len(data_df) - 1
        self.current_step = 0
        self.state = self._get_state(self.current_step).astype(np.float32)

    def normalize_state(self, state):
        max_values = np.array([
            100, 100, 1e12, 1e12, 1e10, 1e10, 4e3, 64*1e9, 100, 1e6, 1e6, 1e6, 1e6
        ], dtype=np.float32)
        return state / max_values

    def _get_state(self, step):
        data = self.data_df.iloc[step]
        state = np.array([
            data['cpu_percent'], data['memory_percent'], data['disk_read_bytes'],
            data['disk_write_bytes'], data['network_sent_bytes'], data['network_recv_bytes'],
            data['cpu_freq'] if pd.notna(data['cpu_freq']) else 0, data['available_memory'],
            data['swap_memory_percent'], data['disk_read_time'], data['disk_write_time'],
            data['network_packets_sent'], data['network_packets_recv']
        ], dtype=np.float32)
        return self.normalize_state(state)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.current_step = self.max_steps

        cpu_adjust, mem_adjust, disk_io_adjust, net_io_adjust, ctx_adjust, int_adjust = action
        self.state[0] = np.clip(self.state[0] + cpu_adjust * 0.1, 0, 1)
        self.state[1] = np.clip(self.state[1] + mem_adjust * 0.1, 0, 1)
        self.state[2] = np.clip(self.state[2] + disk_io_adjust * 0.05, 0, 1)
        self.state[3] = np.clip(self.state[3] + disk_io_adjust * 0.05, 0, 1)
        self.state[4] = np.clip(self.state[4] + net_io_adjust * 0.05, 0, 1)
        self.state[5] = np.clip(self.state[5] + net_io_adjust * 0.05, 0, 1)

        new_data = self._get_state(self.current_step).astype(np.float32)
        self.state[6:] = new_data[6:]

        reward = self._get_reward()
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self._get_state(self.current_step).astype(np.float32)
        return self.state, {}

    def _get_reward(self):
        cpu_usage = self.state[0] * 100
        memory_usage = self.state[1] * 100
        disk_io = (self.state[2] + self.state[3])
        network_io = (self.state[4] + self.state[5])

        reward = 0
        if 40 <= cpu_usage <= 60:
            reward += 1
        else:
            reward -= abs(cpu_usage - 50) / 20

        if 40 <= memory_usage <= 60:
            reward += 1
        else:
            reward -= abs(memory_usage - 50) / 20

        reward -= disk_io / 1e9
        reward -= network_io / 1e9

        if cpu_usage < 20 or cpu_usage > 80:
            reward -= 2
        if memory_usage < 20 or memory_usage > 80:
            reward -= 2

        reward -= np.sum(np.abs(self.state[:6] - self._get_state(self.current_step)[:6])) * 0.05

        return reward

# Dialog for RL Metrics
class RLTrainingMetricsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RL Training Metrics")
        self.layout = QVBoxLayout(self)

        self.figure_rl, self.ax_rl = plt.subplots()
        self.canvas_rl = FigureCanvas(self.figure_rl)
        self.layout.addWidget(self.canvas_rl)

        self.details_label = QLabel("Training Details:")
        self.layout.addWidget(self.details_label)

    def update_metrics(self, rewards):
        self.ax_rl.clear()
        self.ax_rl.plot(rewards, label='Rewards')
        self.ax_rl.set_xlabel('Episodes')
        self.ax_rl.set_ylabel('Rewards')
        self.ax_rl.set_title('RL Training Metrics')
        self.ax_rl.legend()
        self.ax_rl.grid(True)
        self.canvas_rl.draw()

    def update_details(self, details):
        self.details_label.setText(f"Training Details:\n{details}")

# Dialog for LSTM Predictions
class LSTMPredictionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LSTM Predictions")
        self.layout = QVBoxLayout(self)

        self.figure_lstm, self.axes_lstm = plt.subplots(6, 1, figsize=(10, 12))
        self.canvas_lstm = FigureCanvas(self.figure_lstm)
        self.layout.addWidget(self.canvas_lstm)

    def update_predictions(self, actual, predicted):
        feature_names = ['CPU %', 'Memory %', 'Disk Read Bytes', 'Disk Write Bytes', 'Network Sent Bytes', 'Network Recv Bytes']
        for i, ax in enumerate(self.axes_lstm):
            ax.clear()
            ax.plot(actual[:, i], label='Actual')
            ax.plot(predicted[:, i], label='Predicted')
            ax.set_title(f'LSTM Predictions for {feature_names[i]}')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True)
        self.figure_lstm.tight_layout()
        self.canvas_lstm.draw()

# Main GUI class
class SystemMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = []
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.recording = False
        self.lstm_model = self.init_lstm_model()
        self.rl_metrics_dialog = RLTrainingMetricsDialog(self)
        self.lstm_predictions_dialog = LSTMPredictionsDialog(self)

    def initUI(self):
        self.setWindowTitle('System Monitor')

        self.start_btn = QPushButton('Start Recording', self)
        self.start_btn.clicked.connect(self.start_recording)

        self.stop_btn = QPushButton('Stop Recording', self)
        self.stop_btn.clicked.connect(self.stop_recording)
        
        self.pause_btn = QPushButton('Pause Recording', self)
        self.pause_btn.clicked.connect(self.pause_recording)
        
        self.resume_btn = QPushButton('Resume Recording', self)
        self.resume_btn.clicked.connect(self.resume_recording)

        self.train_rl_btn = QPushButton('Train RL Model', self)
        self.train_rl_btn.clicked.connect(self.train_rl_model)

        self.train_lstm_btn = QPushButton('Train LSTM Model', self)
        self.train_lstm_btn.clicked.connect(self.train_lstm_model)

        self.predict_lstm_btn = QPushButton('Predict with LSTM Model', self)
        self.predict_lstm_btn.clicked.connect(self.predict_lstm)

        self.export_btn = QPushButton('Export Data', self)
        self.export_btn.clicked.connect(self.export_data)

        layout = QVBoxLayout()
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.resume_btn)
        layout.addWidget(self.train_rl_btn)
        layout.addWidget(self.train_lstm_btn)
        layout.addWidget(self.predict_lstm_btn)
        layout.addWidget(self.export_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.figure_proc, self.ax_proc = plt.subplots()
        self.canvas_proc = FigureCanvas(self.figure_proc)
        layout.addWidget(self.canvas_proc)

        self.show()

    def start_recording(self):
        self.recording = True
        self.timer.start(1000)  # 1 second interval

    def stop_recording(self):
        self.timer.stop()
        self.recording = False

    def pause_recording(self):
        if self.recording:
            self.timer.stop()

    def resume_recording(self):
        if self.recording:
            self.timer.start(1000)

    def update_data(self):
        data = collect_data()
        self.data.append(data)
        self.plot_data()
        self.plot_top_processes()
        self.check_alerts(data)

    def plot_data(self):
        df = pd.DataFrame(self.data)
        self.ax.clear()
        self.ax.plot(df['timestamp'], df['cpu_percent'], label='CPU %')
        self.ax.plot(df['timestamp'], df['memory_percent'], label='Memory %')
        self.ax.legend()
        self.canvas.draw()

    def plot_top_processes(self):
        processes = [(proc.info['name'], proc.info['cpu_percent'], proc.info['memory_percent']) for proc in psutil.process_iter(attrs=['name', 'cpu_percent', 'memory_percent'])]
        processes = [(name, cpu, mem) for name, cpu, mem in processes if cpu is not None and mem is not None]
        processes = sorted(processes, key=lambda x: x[1], reverse=True)[:5]
        names, cpu_percents, mem_percents = zip(*processes) if processes else ([], [], [])

        self.ax_proc.clear()
        self.ax_proc.barh(names, cpu_percents, color='blue', label='CPU %')
        self.ax_proc.barh(names, mem_percents, color='orange', label='Memory %', left=cpu_percents)
        self.ax_proc.set_xlabel('Usage Percent')
        self.ax_proc.set_title('Top Processes by CPU and Memory Usage')
        self.ax_proc.legend()
        self.canvas_proc.draw()

    def check_alerts(self, data):
        if data['cpu_percent'] > 80:
            self.show_message("High CPU Usage Alert", f"CPU usage is at {data['cpu_percent']}%")
        if data['memory_percent'] > 80:
            self.show_message("High Memory Usage Alert", f"Memory usage is at {data['memory_percent']}%")

    def train_rl_model(self):
        data_df = pd.DataFrame(self.data)
        env = EnhancedComplexSystemEnv(data_df)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save("ppo_complex_system_env")

        # Dummy rewards for demonstration purposes
        rewards = np.abs(np.random.randn(100))
        self.rl_metrics_dialog.update_metrics(rewards)
        self.rl_metrics_dialog.update_details("Total timesteps: 100000\nPolicy: MlpPolicy\nEnv: EnhancedComplexSystemEnv")
        self.rl_metrics_dialog.show()

        self.show_message("RL Model Training", "RL Model training is complete!")

    def train_lstm_model(self):
        data_df = pd.DataFrame(self.data)
        data_values = data_df[['cpu_percent', 'memory_percent', 'disk_read_bytes', 'disk_write_bytes', 'network_sent_bytes', 'network_recv_bytes']].values
        dataset = SystemDataDataset(data_values)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.lstm_model.train()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        for epoch in range(10):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                self.lstm_model.hidden_cell = (torch.zeros(1, 1, self.lstm_model.hidden_layer_size),
                                               torch.zeros(1, 1, self.lstm_model.hidden_layer_size))
                y_pred = self.lstm_model(inputs)
                single_loss = loss_function(y_pred, targets)
                single_loss.backward()
                optimizer.step()

        self.show_message("LSTM Model Training", "LSTM Model training is complete!")
    
    def predict_lstm(self):
        data_df = pd.DataFrame(self.data)
        data_values = data_df[['cpu_percent', 'memory_percent', 'disk_read_bytes', 'disk_write_bytes', 'network_sent_bytes', 'network_recv_bytes']].values
        input_seq = torch.tensor(data_values[-1:], dtype=torch.float32)
        
        self.lstm_model.eval()
        with torch.no_grad():
            self.lstm_model.hidden_cell = (torch.zeros(1, 1, self.lstm_model.hidden_layer_size),
                                           torch.zeros(1, 1, self.lstm_model.hidden_layer_size))
            prediction = self.lstm_model(input_seq).unsqueeze(0)
        
        actual = data_values[-10:]  # Last 10 actual values for comparison
        predicted = np.vstack((data_values[-9:], prediction.numpy()))
        
        self.lstm_predictions_dialog.update_predictions(actual, predicted)
        self.lstm_predictions_dialog.show()

    def init_lstm_model(self):
        input_size = 6  # Number of features in the input data
        hidden_layer_size = 100
        output_size = 6  # Number of features in the output data
        return LSTMModel(input_size, hidden_layer_size, output_size)

    def show_message(self, title, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def export_data(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Data", "", 
                                                   "CSV Files (*.csv);;JSON Files (*.json);;XML Files (*.xml);;Excel Files (*.xlsx);;HDF5 Files (*.h5)", 
                                                   options=options)
        if file_name:
            df = pd.DataFrame(self.data)
            if file_name.endswith('.csv'):
                df.to_csv(file_name, index=False)
            elif file_name.endswith('.json'):
                df.to_json(file_name, orient='records', lines=True)
            elif file_name.endswith('.xml'):
                df.to_xml(file_name)
            elif file_name.endswith('.xlsx'):
                df.to_excel(file_name, index=False)
            elif file_name.endswith('.h5'):
                df.to_hdf(file_name, key='df', mode='w')
            self.show_message("Export Data", "Data export is complete!")

# Main Execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SystemMonitor()
    sys.exit(app.exec_())
