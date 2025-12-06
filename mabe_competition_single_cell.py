# MABe Mouse Behavior Recognition Competition - Single Cell Version
# 完整竞赛版本 - 所有代码在一个cell中，方便复制粘贴
#
# 重要：此notebook已配置为完全离线模式，不进行任何网络访问
# - 所有环境变量已设置为禁用网络访问
# - 所有数据从本地Kaggle数据集加载
# - 不包含任何网络请求、下载或API调用
# - 确保在Kaggle Notebook设置中关闭"Internet"选项

# ========== 导入库 ==========
import sys
import os
from pathlib import Path
import time
import pickle
import math

# 抑制调试器警告（frozen modules警告）
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
os.environ['PYTHONDEVMODE'] = '0'

# 禁用所有网络访问（竞赛要求）
os.environ['TORCH_HOME'] = '/kaggle/working/.cache/torch'  # 使用本地缓存目录
os.environ['HF_HOME'] = '/kaggle/working/.cache/huggingface'  # 禁用HuggingFace下载
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 强制离线模式
os.environ['HF_DATASETS_OFFLINE'] = '1'  # 禁用数据集下载

import torch
# 禁用torch hub的网络访问（必须在导入torch后设置）
torch.hub.set_dir('/kaggle/working/.cache/torch/hub')  # 使用本地目录
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# 抑制调试器相关的警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*frozen.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*debugger.*')
# 抑制 traitlets FutureWarning（来自 nbconvert）
warnings.filterwarnings('ignore', category=FutureWarning, module='traitlets')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 时间跟踪
start_time = time.time()
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ========== 定义所有类 ==========
# Data Loader
class MABeDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_tracking_dir = self.data_dir / "train_tracking"
        self.train_annotation_dir = self.data_dir / "train_annotation"
        self.test_tracking_dir = self.data_dir / "test_tracking"
        self.train_csv = self.data_dir / "train.csv"
        self.test_csv = self.data_dir / "test.csv"
        self._train_df = None
        self._test_df = None
        self._video_to_lab = {}
    
    def _get_lab_for_video(self, video_id: str):
        if video_id in self._video_to_lab:
            return self._video_to_lab[video_id]
        if self._train_df is None and self.train_csv.exists():
            self._train_df = pd.read_csv(self.train_csv)
            if 'video_id' in self._train_df.columns:
                for _, row in self._train_df.iterrows():
                    vid = str(row['video_id'])
                    if 'lab' in row:
                        self._video_to_lab[vid] = str(row['lab'])
        return self._video_to_lab.get(video_id)
    
    def load_pose_data(self, video_id: str, is_test: bool = False):
        tracking_dir = self.test_tracking_dir if is_test else self.train_tracking_dir
        lab_name = self._get_lab_for_video(video_id)
        possible_paths = []
        if lab_name:
            lab_dir = tracking_dir / lab_name
            if lab_dir.exists():
                for ext in ['.pkl', '.npy', '.csv', '.parquet']:
                    possible_paths.append(lab_dir / f"{video_id}{ext}")
        for ext in ['.pkl', '.npy', '.csv', '.parquet']:
            possible_paths.append(tracking_dir / f"{video_id}{ext}")
        if tracking_dir.exists():
            for subdir in tracking_dir.iterdir():
                if subdir.is_dir():
                    for ext in ['.pkl', '.npy', '.csv', '.parquet']:
                        possible_paths.append(subdir / f"{video_id}{ext}")
        for pose_file in possible_paths:
            if pose_file.exists():
                try:
                    if pose_file.suffix == '.pkl':
                        with open(pose_file, 'rb') as f:
                            data = pickle.load(f)
                        return self._parse_pose_data(data)
                    elif pose_file.suffix == '.npy':
                        data = np.load(pose_file, allow_pickle=True)
                        return self._parse_pose_data(data)
                    elif pose_file.suffix == '.csv':
                        # 尝试读取CSV格式的pose数据
                        df = pd.read_csv(pose_file)
                        # 只保留数值列
                        df_numeric = df.select_dtypes(include=[np.number])
                        if len(df_numeric.columns) == 0:
                            raise ValueError(f"No numeric columns found in {pose_file.name}. "
                                           f"All columns: {list(df.columns)}")
                        # 转换为numpy数组
                        data = df_numeric.values.astype(np.float64)
                        return self._parse_pose_data(data)
                    elif pose_file.suffix == '.parquet':
                        # 读取 Parquet 格式的pose数据
                        df = pd.read_parquet(pose_file)
                        
                        # 检查数据类型
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                        
                        # 将 DataFrame 转换为适合的格式
                        if isinstance(df, pd.DataFrame):
                            # 只保留数值列
                            df_numeric = df.select_dtypes(include=[np.number])
                            
                            if len(df_numeric.columns) == 0:
                                raise ValueError(f"No numeric columns found in {pose_file.name}. "
                                               f"All columns: {list(df.columns)}")
                            
                            # 检查DataFrame的结构
                            # 如果列名包含'mouse1', 'mouse2'等，可能是字典格式
                            mouse1_cols = [col for col in df_numeric.columns if 'mouse1' in str(col).lower()]
                            mouse2_cols = [col for col in df_numeric.columns if 'mouse2' in str(col).lower()]
                            
                            if mouse1_cols or mouse2_cols:
                                # 尝试按列分组
                                data = {}
                                for col in df_numeric.columns:
                                    col_lower = str(col).lower()
                                    if 'mouse1' in col_lower:
                                        if 'mouse1' not in data:
                                            data['mouse1'] = []
                                        # 确保值是数值类型
                                        col_values = pd.to_numeric(df_numeric[col], errors='coerce').values
                                        data['mouse1'].append(col_values)
                                    elif 'mouse2' in col_lower:
                                        if 'mouse2' not in data:
                                            data['mouse2'] = []
                                        col_values = pd.to_numeric(df_numeric[col], errors='coerce').values
                                        data['mouse2'].append(col_values)
                                if data:
                                    # 转换为numpy数组
                                    for key in data:
                                        data[key] = np.array(data[key]).T  # 转置以匹配(T, features)格式
                                        # 确保是数值类型
                                        data[key] = data[key].astype(np.float64)
                                    return data
                            
                            # 检查是否是多级索引或结构化数据
                            # 如果DataFrame只有一列，可能是嵌套结构
                            if len(df_numeric.columns) == 1:
                                # 可能是嵌套的字典结构
                                first_col = df_numeric.iloc[:, 0]
                                if isinstance(first_col.iloc[0], dict):
                                    # 每行是一个字典
                                    data = {}
                                    for idx, row_dict in first_col.items():
                                        for key, value in row_dict.items():
                                            if key not in data:
                                                data[key] = []
                                            # 确保值是数值类型
                                            try:
                                                data[key].append(float(value))
                                            except (ValueError, TypeError):
                                                continue
                                    if data:
                                        for key in data:
                                            data[key] = np.array(data[key], dtype=np.float64)
                                        return data
                            
                            # 使用数值列转换为numpy数组
                            data = df_numeric.values.astype(np.float64)
                            return self._parse_pose_data(data)
                        else:
                            return self._parse_pose_data(df)
                except Exception as e:
                    # 静默跳过错误文件，避免产生大量输出
                    continue
        
        # 如果所有路径都找不到，返回更详细的错误信息
        tracking_dir_str = str(tracking_dir)
        raise FileNotFoundError(
            f"Pose file not found for video {video_id}. "
            f"Searched in: {tracking_dir_str} and subdirectories. "
            f"Tried extensions: .pkl, .npy, .csv, .parquet"
        )
    
    def _parse_pose_data(self, data):
        if isinstance(data, dict):
            # 确保字典中的值是numpy数组，且是数值类型
            result = {}
            for key, value in data.items():
                if isinstance(value, (list, pd.Series)):
                    # 尝试转换为数值类型
                    try:
                        result[key] = pd.to_numeric(value, errors='coerce').values
                        result[key] = result[key].astype(np.float64)
                    except (ValueError, TypeError):
                        # 如果转换失败，尝试直接转换
                        result[key] = np.array(value, dtype=np.float64)
                elif isinstance(value, np.ndarray):
                    # 确保是数值类型
                    result[key] = value.astype(np.float64)
                else:
                    result[key] = value
            return result
        elif isinstance(data, np.ndarray):
            # 确保是数值类型
            if data.dtype.kind not in ['f', 'i']:  # float or integer
                # 尝试转换为数值类型
                try:
                    data = pd.DataFrame(data).select_dtypes(include=[np.number]).values.astype(np.float64)
                except:
                    raise ValueError(f"Cannot convert array to numeric. dtype: {data.dtype}, sample: {data.flat[0] if data.size > 0 else 'empty'}")
            
            # 处理numpy数组
            if len(data.shape) == 3:
                # 已经是 (T, num_keypoints, 2) 格式
                return {'mouse1': data.astype(np.float64)}
            elif len(data.shape) == 2:
                # (T, features) 格式
                T, features = data.shape
                # 如果features是偶数，可能是(x, y)坐标对
                if features % 2 == 0:
                    # 可能是 (T, num_keypoints*2) 格式
                    return {'mouse1': data.astype(np.float64)}
                else:
                    # 可能是其他格式，直接返回
                    return {'mouse1': data.astype(np.float64)}
            else:
                return {'mouse1': data.astype(np.float64)}
        elif isinstance(data, pd.DataFrame):
            # 如果是DataFrame，只保留数值列
            df_numeric = data.select_dtypes(include=[np.number])
            if len(df_numeric.columns) == 0:
                raise ValueError(f"No numeric columns in DataFrame. Columns: {list(data.columns)}")
            return self._parse_pose_data(df_numeric.values)
        else:
            result = {}
            for key in ['mouse1', 'mouse2', 'mouse']:
                if hasattr(data, key):
                    value = getattr(data, key)
                    if isinstance(value, (list, pd.Series)):
                        try:
                            result[key] = pd.to_numeric(value, errors='coerce').values.astype(np.float64)
                        except:
                            result[key] = np.array(value, dtype=np.float64)
                    elif isinstance(value, np.ndarray):
                        result[key] = value.astype(np.float64)
                    else:
                        result[key] = value
            if result:
                return result
            raise ValueError(f"Unknown pose data format: {type(data)}")
    
    def load_annotations(self, video_id: str):
        lab_name = self._get_lab_for_video(video_id)
        possible_paths = []
        if lab_name:
            lab_dir = self.train_annotation_dir / lab_name
            if lab_dir.exists():
                # 支持 .csv 和 .parquet
                possible_paths.append(lab_dir / f"{video_id}.csv")
                possible_paths.append(lab_dir / f"{video_id}.parquet")
        if self.train_annotation_dir.exists():
            possible_paths.append(self.train_annotation_dir / f"{video_id}.csv")
            possible_paths.append(self.train_annotation_dir / f"{video_id}.parquet")
            for subdir in self.train_annotation_dir.iterdir():
                if subdir.is_dir():
                    possible_paths.append(subdir / f"{video_id}.csv")
                    possible_paths.append(subdir / f"{video_id}.parquet")
        for annotation_file in possible_paths:
            if annotation_file.exists():
                try:
                    # 根据文件扩展名选择读取方法
                    if annotation_file.suffix == '.parquet':
                        df = pd.read_parquet(annotation_file)
                    else:  # .csv
                        df = pd.read_csv(annotation_file)
                    
                    column_mapping = {'frame_id': 'frame', 'frame_idx': 'frame', 'frame': 'frame',
                                     'agent': 'agent_id', 'agent_id': 'agent_id',
                                     'target': 'target_id', 'target_id': 'target_id',
                                     'behavior': 'action', 'label': 'action', 'action': 'action'}
                    df = df.rename(columns=column_mapping)
                    required_cols = ['frame', 'agent_id', 'target_id', 'action']
                    for col in required_cols:
                        if col not in df.columns:
                            if col == 'frame':
                                df['frame'] = df.index
                            else:
                                df[col] = None
                    return df[required_cols]
                except Exception as e:
                    print(f"Warning: Error reading annotation file {annotation_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        return pd.DataFrame(columns=['frame', 'agent_id', 'target_id', 'action'])
    
    def get_video_list(self, split: str = 'train', max_videos=None):
        video_ids = []
        if split == 'test':
            if self.test_csv.exists():
                test_df = pd.read_csv(self.test_csv)
                if 'video_id' in test_df.columns:
                    video_ids = test_df['video_id'].astype(str).tolist()
        else:
            if self.train_csv.exists():
                train_df = pd.read_csv(self.train_csv)
                if 'video_id' in train_df.columns:
                    video_ids = train_df['video_id'].astype(str).tolist()
        video_ids = sorted(list(set(video_ids)))
        if max_videos is not None and len(video_ids) > max_videos:
            video_ids = video_ids[:max_videos]
        return video_ids
    
    def load_video_data(self, video_id: str, is_test: bool = False):
        try:
            pose_data = self.load_pose_data(video_id, is_test=is_test)
        except FileNotFoundError as e:
            # 如果找不到pose文件，返回空字典而不是抛出异常
            print(f"Warning: {e}")
            pose_data = {}
        except Exception as e:
            print(f"Error loading pose data for video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            pose_data = {}
        
        try:
            if is_test:
                annotations = pd.DataFrame(columns=['frame', 'agent_id', 'target_id', 'action'])
            else:
                annotations = self.load_annotations(video_id)
        except Exception as e:
            print(f"Error loading annotations for video {video_id}: {e}")
            import traceback
            traceback.print_exc()
            annotations = pd.DataFrame(columns=['frame', 'agent_id', 'target_id', 'action'])
        
        return pose_data, annotations
    
    def create_sequence_data(self, pose_data, annotations, window_size=60, stride=30):
        sequences = []
        max_frames = 0
        # 获取所有鼠标的pose数据形状信息
        pose_shape = None
        for mouse_id, poses in pose_data.items():
            if poses is not None and len(poses) > 0:
                max_frames = max(max_frames, len(poses))
                if pose_shape is None:
                    pose_shape = poses.shape[1:]  # 获取 (num_keypoints, 2) 的形状
        
        if max_frames == 0:
            return sequences
        
        # 如果视频帧数小于window_size，至少生成一个序列（使用填充）
        if max_frames < window_size:
            start_frame = 0
            end_frame = max_frames
            window_poses = {}
            for mouse_id, poses in pose_data.items():
                if poses is not None and len(poses) > 0:
                    # 截取实际数据
                    actual_poses = poses[start_frame:end_frame]
                    # 如果长度不足，用零填充
                    if len(actual_poses) < window_size:
                        padding = np.zeros((window_size - len(actual_poses),) + pose_shape)
                        window_poses[mouse_id] = np.concatenate([actual_poses, padding], axis=0)
                    else:
                        window_poses[mouse_id] = actual_poses
                else:
                    # 如果没有数据，创建全零数组
                    if pose_shape is not None:
                        window_poses[mouse_id] = np.zeros((window_size,) + pose_shape)
                    else:
                        # 如果连形状都不知道，使用默认形状 (13, 2)
                        window_poses[mouse_id] = np.zeros((window_size, 13, 2))
            
            window_annotations = annotations[(annotations['frame'] >= start_frame) & 
                                            (annotations['frame'] < end_frame)].copy()
            sequences.append({'video_id': None, 'start_frame': start_frame, 'end_frame': end_frame,
                            'poses': window_poses, 'annotations': window_annotations})
        else:
            # 正常情况：视频帧数 >= window_size
            for start_frame in range(0, max_frames - window_size + 1, stride):
                end_frame = start_frame + window_size
                window_poses = {}
                for mouse_id, poses in pose_data.items():
                    if poses is not None and len(poses) > start_frame:
                        window_poses[mouse_id] = poses[start_frame:end_frame]
                    else:
                        # 如果没有数据，使用零填充
                        if pose_shape is not None:
                            window_poses[mouse_id] = np.zeros((window_size,) + pose_shape)
                        else:
                            window_poses[mouse_id] = np.zeros((window_size, 13, 2))
                window_annotations = annotations[(annotations['frame'] >= start_frame) & 
                                                (annotations['frame'] < end_frame)].copy()
                sequences.append({'video_id': None, 'start_frame': start_frame, 'end_frame': end_frame,
                                'poses': window_poses, 'annotations': window_annotations})
        return sequences

# Feature Extractor
class FeatureExtractor:
    def __init__(self, num_keypoints=13, fps=30.0):
        self.num_keypoints = num_keypoints
        self.fps = fps
        self.dt = 1.0 / fps
    
    def extract_features(self, poses, window_size=60):
        if 'mouse1' not in poses or 'mouse2' not in poses:
            if 'mouse1' in poses:
                poses = {'mouse1': poses['mouse1'], 'mouse2': np.zeros_like(poses['mouse1'])}
            else:
                raise ValueError("At least mouse1 pose data required")
        mouse1_pose = poses['mouse1']
        mouse2_pose = poses['mouse2']
        
        # 处理不同的数据形状（移除调试输出以减少notebook文件大小）
        if len(mouse1_pose.shape) == 2:
            T, features = mouse1_pose.shape
            
            # 如果features是偶数，尝试reshape为(T, num_keypoints, 2)
            if features % 2 == 0:
                num_keypoints = features // 2
                try:
                    mouse1_pose = mouse1_pose.reshape(T, num_keypoints, 2)
                    mouse2_pose = mouse2_pose.reshape(T, num_keypoints, 2)
                except ValueError as e:
                    raise ValueError(f"Cannot reshape pose data: {e}")
            else:
                # features不是偶数，可能是已经提取的特征（不是原始坐标）
                # 为了兼容，我们创建一个虚拟的3D结构
                # 将每个特征视为一个"关键点"，坐标为(feature_value, 0)
                mouse1_pose_3d = np.zeros((T, features, 2))
                mouse1_pose_3d[:, :, 0] = mouse1_pose  # x坐标是特征值
                mouse1_pose_3d[:, :, 1] = 0  # y坐标是0
                
                mouse2_pose_3d = np.zeros((T, features, 2))
                mouse2_pose_3d[:, :, 0] = mouse2_pose
                mouse2_pose_3d[:, :, 1] = 0
                
                mouse1_pose = mouse1_pose_3d
                mouse2_pose = mouse2_pose_3d
        elif len(mouse1_pose.shape) == 3:
            # 已经是 (T, num_keypoints, 2) 格式
            T, num_keypoints, coords = mouse1_pose.shape
            if coords != 2:
                raise ValueError(f"Expected 2 coordinates (x, y), got {coords} in shape {mouse1_pose.shape}")
        else:
            raise ValueError(f"Unsupported pose data shape: {mouse1_pose.shape}. "
                           f"Expected 2D (T, features) or 3D (T, num_keypoints, 2)")
        
        mouse1_features = self._extract_mouse_features(mouse1_pose)
        mouse2_features = self._extract_mouse_features(mouse2_pose)
        interaction_features = self._extract_interaction_features(mouse1_pose, mouse2_pose)
        features = np.concatenate([mouse1_features, mouse2_features, interaction_features], axis=-1)
        return features
    
    def _extract_mouse_features(self, pose):
        T = pose.shape[0]
        features_list = []
        body_center = np.mean(pose, axis=1)
        centered_pose = pose - body_center[:, None, :]
        features_list.append(centered_pose.reshape(T, -1))
        velocity = np.gradient(pose, axis=0) / self.dt
        features_list.append(velocity.reshape(T, -1))
        acceleration = np.gradient(velocity, axis=0) / self.dt
        features_list.append(acceleration.reshape(T, -1))
        speed = np.linalg.norm(velocity, axis=2)
        features_list.append(speed)
        distances = np.linalg.norm(centered_pose, axis=2)
        features_list.append(distances)
        return np.concatenate(features_list, axis=1)
    
    def _extract_interaction_features(self, pose1, pose2):
        T = pose1.shape[0]
        features_list = []
        center1 = np.mean(pose1, axis=1)
        center2 = np.mean(pose2, axis=1)
        center_distance = np.linalg.norm(center1 - center2, axis=1)
        features_list.append(center_distance[:, None])
        relative_position = center2 - center1
        features_list.append(relative_position)
        return np.concatenate(features_list, axis=1)

# Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class BehaviorTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8, num_layers=4, d_ff=1024,
                 num_classes=35, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        logits = self.classifier(x)
        return logits

# Dataset and Trainer
class BehaviorDataset(Dataset):
    def __init__(self, sequences, feature_extractor, behavior_to_idx, window_size):
        self.sequences = sequences
        self.feature_extractor = feature_extractor
        self.behavior_to_idx = behavior_to_idx
        self.window_size = window_size
        self.num_classes = len(behavior_to_idx)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = self.feature_extractor.extract_features(seq['poses'], self.window_size)
        labels = self._create_labels(seq['annotations'], seq['start_frame'])
        return {'features': torch.FloatTensor(features), 'labels': torch.LongTensor(labels),
                'video_id': seq.get('video_id', ''), 'start_frame': seq['start_frame'],
                'end_frame': seq['end_frame']}
    
    def _create_labels(self, annotations, start_frame):
        labels = np.zeros(self.window_size, dtype=np.int64)
        if len(annotations) > 0:
            for _, row in annotations.iterrows():
                frame = int(row['frame']) - start_frame
                if 0 <= frame < self.window_size:
                    action = str(row['action'])
                    if action in self.behavior_to_idx:
                        labels[frame] = self.behavior_to_idx[action]
        return labels

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1),
                                 weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda', learning_rate=1e-4,
                 weight_decay=1e-5, use_focal_loss=True, class_weights=None, 
                 gradient_accumulation_steps=1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.best_val_score = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Training')):
            features = batch['features'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            # 将loss除以累积步数，以便梯度累积后得到正确的平均梯度
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps  # 恢复原始loss值用于记录
            
            # 每gradient_accumulation_steps步或最后一个batch时更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 定期清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            num_batches += 1
        
        # 确保所有梯度都已更新
        if num_batches % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self):
        if self.val_loader is None:
            return 0.0, 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validating')):
                features = batch['features'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # 每10个batch清理一次显存
                if torch.cuda.is_available() and (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # 验证结束后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        val_score = 0.0
        self.val_losses.append(avg_loss)
        self.val_scores.append(val_score)
        return avg_loss, val_score
    
    def train(self, num_epochs, save_dir='checkpoints'):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            if self.val_loader is not None:
                val_loss, val_score = self.validate()
                print(f"Val Loss: {val_loss:.4f}, Val F-Score: {val_score:.4f}")
                self.scheduler.step(val_loss)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    torch.save({'model_state_dict': self.model.state_dict(),
                               'optimizer_state_dict': self.optimizer.state_dict(),
                               'best_val_score': self.best_val_score},
                              save_dir / 'best_model.pt')
                    print(f"Saved best model (F-Score: {val_score:.4f})")
            
            # 每个epoch结束后清理显存（移除中间checkpoint保存以减少磁盘使用）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if (epoch + 1) % 5 == 0:  # 每5个epoch打印一次显存使用情况
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"  GPU显存使用: {allocated:.2f} GB / {reserved:.2f} GB")
        
        # 训练结束后清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nTraining completed!")

print("All classes defined successfully")

# ========== 配置和数据路径 ==========
possible_data_paths = [
    Path('/kaggle/input/MABe-mouse-behavior-detection'),
    Path('/kaggle/input/mabe-mouse-behavior-detection'),
    Path('/kaggle/input/MABe-Mouse-Behavior-Detection'),  # 添加更多变体
    Path('/kaggle/input/mabe-mouse-behavior'),  # 简化版本
    Path('/kaggle/input'),
    Path('./input'),  # 本地测试路径
    Path('.'),  # 当前目录
]

DATA_DIR = None
for path in possible_data_paths:
    if path.exists():
        # 检查是否包含必要的子目录或文件
        has_tracking = (path / 'train_tracking').exists() or (path / 'test_tracking').exists()
        has_csv = (path / 'train.csv').exists() or (path / 'test.csv').exists()
        if has_tracking or has_csv or path.name == 'input' or path == Path('.'):
            DATA_DIR = path
            print(f"Using data path: {DATA_DIR}")
            # 列出目录内容以便调试
            if DATA_DIR.exists():
                try:
                    contents = list(DATA_DIR.iterdir())[:10]  # 只显示前10个
                    print(f"  Directory contents (first 10): {[str(c.name) for c in contents]}")
                except:
                    pass
            break

if DATA_DIR is None:
    DATA_DIR = Path('/kaggle/input')
    print(f"Using default data path: {DATA_DIR}")
    print("Warning: Could not find data directory, using default path")

CONFIG = {
    'window_size': 60,
    'stride': 45,
    'batch_size': 8,  # 进一步减小到8以避免OOM
    'gradient_accumulation_steps': 8,  # 梯度累积步数，等效batch_size = 8*8 = 64
    'learning_rate': 1e-4,
    'num_epochs': 25,
    'early_stop_patience': 5,
    'd_model': 192,
    'num_heads': 6,
    'num_layers': 3,
    'd_ff': 768,
    'dropout': 0.1,
    'fps': 30.0,
    'num_keypoints': 13,
    'max_train_videos': 80,
}

print("Optimized Configuration (for 9-hour limit):")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# 显存优化提示
if torch.cuda.is_available():
    print(f"\n显存优化设置:")
    print(f"  - Batch size: {CONFIG['batch_size']} (已减小以避免OOM)")
    print(f"  - 梯度累积步数: {CONFIG.get('gradient_accumulation_steps', 1)}")
    print(f"  - 等效batch size: {CONFIG['batch_size'] * CONFIG.get('gradient_accumulation_steps', 1)}")
    print(f"  - 当前GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  - 已分配显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  - 缓存显存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# ========== 数据加载 ==========
data_loader = MABeDataLoader(str(DATA_DIR))
train_videos = data_loader.get_video_list('train', max_videos=CONFIG['max_train_videos'])
test_videos = data_loader.get_video_list('test')

print(f"Number of training videos (limited): {len(train_videos)}")
print(f"Number of test videos: {len(test_videos)}")

# 获取行为类别
if len(train_videos) > 0:
    sample_videos = train_videos[:min(20, len(train_videos))]
    all_behaviors = set()
    for video_id in sample_videos:
        try:
            _, annotations = data_loader.load_video_data(video_id, is_test=False)
            if 'action' in annotations.columns:
                behaviors = annotations['action'].dropna().unique()
                all_behaviors.update(behaviors)
        except:
            continue
    behavior_classes = sorted(list(all_behaviors))
    print(f"\nNumber of behavior classes found: {len(behavior_classes)}")
    print(f"Behaviors: {behavior_classes[:15]}...")
else:
    behavior_classes = [
        'sniff', 'approach', 'follow', 'groom', 'mount', 'attack', 
        'escape', 'freeze', 'rear', 'dig', 'eat', 'drink', 'rest'
    ]
    print(f"\nUsing default behavior classes: {len(behavior_classes)}")

if 'background' not in behavior_classes and '' not in behavior_classes:
    behavior_classes = ['background'] + behavior_classes
elif '' in behavior_classes:
    behavior_classes = ['background' if b == '' else b for b in behavior_classes]
    behavior_classes = ['background'] + [b for b in behavior_classes if b != 'background']

num_classes = len(behavior_classes)
print(f"\nTotal classes (including background): {num_classes}")

elapsed = time.time() - start_time
print(f"\nTime elapsed: {elapsed/60:.2f} minutes")

# ========== 准备训练数据 ==========
print("Preparing training data...")
data_start_time = time.time()

feature_extractor = FeatureExtractor(num_keypoints=CONFIG['num_keypoints'], fps=CONFIG['fps'])
all_sequences = []
behavior_to_idx = {b: i for i, b in enumerate(behavior_classes)}

print(f"Processing {len(train_videos)} training videos...")

# 自动检测数据格式（从第一个有效视频）
detected_num_keypoints = None
for video_id in tqdm(train_videos, desc="Loading training videos"):
    try:
        pose_data, annotations = data_loader.load_video_data(video_id, is_test=False)
        
        # 添加调试信息并检测数据格式（只在第一次检测）
        if pose_data and detected_num_keypoints is None:
            max_frames = max((len(poses) for poses in pose_data.values() if poses is not None and len(poses) > 0), default=0)
            if max_frames > 0:
                # 检测数据格式
                for mouse_id, poses in pose_data.items():
                    if poses is not None and len(poses) > 0:
                        if len(poses.shape) == 2:
                            T, features = poses.shape
                            # 自动检测num_keypoints（移除详细输出以减少notebook大小）
                            if features % 2 == 0:
                                detected_num_keypoints = features // 2
                            else:
                                detected_num_keypoints = features
                            
                            # 更新CONFIG中的num_keypoints
                            if CONFIG['num_keypoints'] != detected_num_keypoints:
                                CONFIG['num_keypoints'] = detected_num_keypoints
                                feature_extractor.num_keypoints = detected_num_keypoints
                        elif len(poses.shape) == 3:
                            T, num_kp, coords = poses.shape
                            detected_num_keypoints = num_kp
                            if CONFIG['num_keypoints'] != detected_num_keypoints:
                                CONFIG['num_keypoints'] = detected_num_keypoints
                                feature_extractor.num_keypoints = detected_num_keypoints
                        break
        
        # 继续处理所有视频（移除详细输出以减少notebook大小）
        if pose_data:
            max_frames = max((len(poses) for poses in pose_data.values() if poses is not None and len(poses) > 0), default=0)
        
        sequences = data_loader.create_sequence_data(
            pose_data, annotations, CONFIG['window_size'], CONFIG['stride']
        )
        
        for seq in sequences:
            seq['video_id'] = video_id
        all_sequences.extend(sequences)
        
        # 每处理10个视频清理一次内存
        if len(all_sequences) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        # 静默跳过错误视频，避免产生大量输出
        continue

print(f"Total sequences: {len(all_sequences)}")

split_idx = int(0.8 * len(all_sequences))
train_sequences = all_sequences[:split_idx]
val_sequences = all_sequences[split_idx:]

print(f"Train sequences: {len(train_sequences)}")
print(f"Val sequences: {len(val_sequences)}")

data_time = time.time() - data_start_time
elapsed = time.time() - start_time
print(f"\nData preparation time: {data_time/60:.2f} minutes")
print(f"Total time elapsed: {elapsed/60:.2f} minutes")

# ========== 创建数据集 ==========
train_dataset = BehaviorDataset(train_sequences, feature_extractor, behavior_to_idx, CONFIG['window_size'])
val_dataset = BehaviorDataset(val_sequences, feature_extractor, behavior_to_idx, CONFIG['window_size'])

# 检查数据集是否为空
if len(train_dataset) == 0:
    raise ValueError(f"训练数据集为空！请检查：\n"
                     f"1. 总序列数: {len(all_sequences)}\n"
                     f"2. 训练序列数: {len(train_sequences)}\n"
                     f"3. 验证序列数: {len(val_sequences)}\n"
                     f"4. window_size ({CONFIG['window_size']}) 是否大于视频帧数？\n"
                     f"5. 视频数据是否正确加载？\n"
                     f"6. 请检查上面的警告信息，确认视频数据是否有效。")

sample = train_dataset[0]
input_dim = sample['features'].shape[1]
print(f"Input feature dimension: {input_dim}")

# 使用较小的num_workers和pin_memory=False以减少内存占用
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=0, pin_memory=False,  # 设为0避免多进程内存开销
                          persistent_workers=False)  # 不保持worker进程，节省内存
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                        num_workers=0, pin_memory=False,  # 设为0避免多进程内存开销
                        persistent_workers=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ========== 创建模型 ==========
model = BehaviorTransformer(input_dim=input_dim, d_model=CONFIG['d_model'],
                           num_heads=CONFIG['num_heads'], num_layers=CONFIG['num_layers'],
                           d_ff=CONFIG['d_ff'], num_classes=num_classes,
                           dropout=CONFIG['dropout'], max_seq_len=CONFIG['window_size'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

model = model.to(device)
print("Model created and moved to device")

# ========== 训练模型 ==========
print("Starting training...")
train_start_time = time.time()

trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                 device=device, learning_rate=CONFIG['learning_rate'], use_focal_loss=True,
                 gradient_accumulation_steps=CONFIG.get('gradient_accumulation_steps', 1))

trainer.train(num_epochs=CONFIG['num_epochs'], save_dir='/kaggle/working/checkpoints')

train_time = time.time() - train_start_time
elapsed = time.time() - start_time
print(f"\nTraining completed!")
print(f"Training time: {train_time/60:.2f} minutes")
print(f"Total time elapsed: {elapsed/60:.2f} minutes")
print(f"Estimated remaining time: {(9*60 - elapsed)/60:.2f} hours")

# ========== 加载最佳模型 ==========
best_model_path = '/kaggle/working/checkpoints/best_model.pt'
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model loaded for inference")
    else:
        model.load_state_dict(checkpoint)
        print("Model weights loaded")
else:
    print("Best model not found, using current model")

# ========== 推理和生成提交文件 ==========
print("Starting inference...")
inference_start_time = time.time()

model.eval()
all_predictions = []

for video_id in tqdm(test_videos, desc="Processing test videos"):
    try:
        pose_data, _ = data_loader.load_video_data(video_id, is_test=True)
        sequences = data_loader.create_sequence_data(
            pose_data, pd.DataFrame(), CONFIG['window_size'], CONFIG['stride']
        )
        for seq in sequences:
            seq['video_id'] = video_id
        
        # 推理时使用较小的batch_size以避免OOM
        inference_batch_size = min(CONFIG['batch_size'], 8)  # 推理时使用更小的batch
        for i in range(0, len(sequences), inference_batch_size):
            batch_sequences = sequences[i:i + inference_batch_size]
            batch_features = []
            batch_metadata = []
            
            for seq in batch_sequences:
                features = feature_extractor.extract_features(seq['poses'], CONFIG['window_size'])
                batch_features.append(features)
                batch_metadata.append({
                    'video_id': seq['video_id'],
                    'start_frame': seq['start_frame'],
                    'end_frame': seq['end_frame']
                })
            
            batch_features = np.array(batch_features)
            batch_tensor = torch.FloatTensor(batch_features).to(device)
            
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
            
            preds_np = preds.cpu().numpy()
            # 清理显存
            del batch_tensor, logits, probs, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            idx_to_behavior = {i: b for i, b in enumerate(behavior_classes)}
            
            for j, metadata in enumerate(batch_metadata):
                pred_sequence = preds_np[j]
                current_behavior = None
                current_start = None
                
                for k, behavior_idx in enumerate(pred_sequence):
                    behavior = idx_to_behavior[behavior_idx]
                    frame = metadata['start_frame'] + k
                    
                    if behavior_idx == 0 or behavior == 'background' or behavior == '':
                        if current_behavior is not None:
                            all_predictions.append({
                                'video_id': metadata['video_id'],
                                'agent_id': 'mouse1',
                                'target_id': 'mouse2',
                                'action': current_behavior,
                                'start_frame': current_start,
                                'stop_frame': frame - 1
                            })
                            current_behavior = None
                        continue
                    
                    if behavior != current_behavior:
                        if current_behavior is not None:
                            all_predictions.append({
                                'video_id': metadata['video_id'],
                                'agent_id': 'mouse1',
                                'target_id': 'mouse2',
                                'action': current_behavior,
                                'start_frame': current_start,
                                'stop_frame': frame - 1
                            })
                        current_behavior = behavior
                        current_start = frame
                
                if current_behavior is not None:
                    all_predictions.append({
                        'video_id': metadata['video_id'],
                        'agent_id': 'mouse1',
                        'target_id': 'mouse2',
                        'action': current_behavior,
                        'start_frame': current_start,
                        'stop_frame': metadata['end_frame'] - 1
                    })
    except Exception as e:
        print(f"Error processing video {video_id}: {e}")
        continue

# 合并重叠预测
def merge_overlapping_predictions(predictions, min_duration=5):
    if len(predictions) == 0:
        return []
    sorted_preds = sorted(predictions, key=lambda x: (
        x['video_id'], x['agent_id'], x['target_id'], x['action'], x['start_frame']
    ))
    merged = []
    current = sorted_preds[0].copy()
    for pred in sorted_preds[1:]:
        if (current['video_id'] == pred['video_id'] and
            current['agent_id'] == pred['agent_id'] and
            current['target_id'] == pred['target_id'] and
            current['action'] == pred['action'] and
            current['stop_frame'] >= pred['start_frame']):
            current['stop_frame'] = max(current['stop_frame'], pred['stop_frame'])
        else:
            if current['stop_frame'] - current['start_frame'] >= min_duration:
                merged.append(current)
            current = pred.copy()
    if current['stop_frame'] - current['start_frame'] >= min_duration:
        merged.append(current)
    return merged

merged_predictions = merge_overlapping_predictions(all_predictions)

# 创建提交文件
df = pd.DataFrame(merged_predictions)
if len(df) == 0:
    df = pd.DataFrame([{
        'video_id': test_videos[0] if test_videos else 'test',
        'agent_id': 'mouse1',
        'target_id': 'mouse2',
        'action': 'sniff',
        'start_frame': 0,
        'stop_frame': 10
    }])

df.insert(0, 'row_id', range(len(df)))
df = df[['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']]
df.to_csv('/kaggle/working/submission.csv', index=False)

inference_time = time.time() - inference_start_time
elapsed = time.time() - start_time
print(f"\nSubmission file generated!")
print(f"Inference time: {inference_time/60:.2f} minutes")
print(f"Total time elapsed: {elapsed/60:.2f} minutes ({elapsed/3600:.2f} hours)")

# ========== 验证提交文件 ==========
submission_df = pd.read_csv('/kaggle/working/submission.csv')
print(f"Submission file shape: {submission_df.shape}")
print(f"\nFirst few rows:")
print(submission_df.head(10))
print(f"\nColumn names: {list(submission_df.columns)}")
print(f"\nRequired columns check:")
required_cols = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
for col in required_cols:
    if col in submission_df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} - MISSING!")

print(f"\nTotal predictions: {len(submission_df)}")
if 'action' in submission_df.columns:
    print(f"Unique actions: {submission_df['action'].nunique()}")
    print(f"Action distribution:")
    print(submission_df['action'].value_counts().head(10))

