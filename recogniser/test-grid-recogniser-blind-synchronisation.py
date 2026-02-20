# tests/recogniser/test-blind-synchronisation.py
# loads existing detection pickles, attempts blind synchronisation

import os
import glob
import json
import pickle
import struct
import random
import colorsys
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.signal import correlate
from scipy.optimize import minimize, differential_evolution
from itertools import combinations, permutations
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
from geopy.distance import distance as geopy_distance
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation

# --- Configuration ---
# Update these paths as needed for your environment
GRID_AUDIO_DIR = 'tests/localisation-grid-experiment/taukahara/sample-9mics-5mins'
MAPPING_FILE = 'tests/localisation-grid-experiment/taukahara/micID_to_locationID.json'
LOCATION_FILE = 'tests/localisation-grid-experiment/taukahara/sensor_locations_with_elevation.csv'
PICKLE_DIR = "mutable_detection_data_score0.6" # Folder containing the group_X.pkl files
SAMPLE_RATE = 48000
STFT_PARAMS = {'n_fft': 2048, 'win_length': 2048, 'hop_length': 512}
CONSISTENCY_THRESHOLD_S = 0.005 # 5ms tolerance for cycle errors
NUM_BOOTSTRAP_ROUNDS = 10
GROUPS_PER_ROUND = 50
SKIP_SYNCHRONISATION = True  # load existing offsets and skip optimization
OFFSETS_FILE = "calculated_blind_offsets_averaged.json"
SAVE_LOCALISATION_DATA = True
LOCALISATION_OUTPUT_DIR = 'localisation_results'

SPECIES_DISPLAY_NAMES = {
    'eurgre1': 'European Greenfinch',
    'silver3': 'Tauhou',
    'eurbla': 'Eurasian Blackbird',
    'rinphe1': 'Ring-necked Pheasant',
    'homo_sapien': 'Car',
    'calqua': 'California Quail',
    'whfher1': 'Kōtuku',
    'nezbel1': 'Korimako',
    'mallar3': 'Rakiraki',
    'swahar1': 'Kāhu',
    'soioys1': 'South Island Oystercatcher',
    'gryger1': 'Riroriro',
    'varoys1': 'Variable Oystercatcher',
    'nezpig2': 'Kererū'
}

def parse_guano_metadata(data: bytes) -> dict:
    metadata = {}
    try:
        text = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        return {}
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip().replace('\\n', '\n')
    return metadata

def extract_guano_metadata_with_duration(file_path: str, sample_rate: int) -> dict:
    metadata = {}
    try:
        with open(file_path, 'rb') as f:
            f.seek(4); struct.unpack('<I', f.read(4))[0]; f.seek(36)
            data_chunk_header = f.read(4)
            if data_chunk_header != b'data':
                f.seek(12)
                while f.read(4) != b'data':
                    sub_chunk_size = struct.unpack('<I', f.read(4))[0]
                    f.seek(sub_chunk_size, 1)
            data_size = struct.unpack('<I', f.read(4))[0]
            metadata['duration_s'] = (data_size // 2) / sample_rate
            f.seek(0)
            if f.read(4) != b'RIFF' or f.read(4) == b'' or f.read(4) != b'WAVE': return metadata
            while True:
                chunk_id = f.read(4)
                if not chunk_id: break
                chunk_size = struct.unpack('<I', f.read(4))[0]
                if chunk_id == b'guan':
                    metadata.update(parse_guano_metadata(f.read(chunk_size)))
                    break
                else:
                    f.seek(chunk_size, 1)
                if chunk_size % 2 != 0: f.seek(1, 1)
    except Exception: pass
    return metadata

def get_location_id(device_id: str, timestamp: datetime, location_mapping: dict) -> int | None:
    if device_id not in location_mapping: return None
    device_dates = location_mapping[device_id]['dates_locationIDs']
    recording_mmdd = timestamp.strftime('%m%d')
    relevant_dates = [d for d in device_dates.keys() if d <= recording_mmdd]
    if not relevant_dates: return None
    return device_dates[max(relevant_dates)]

def resample_log_mask_to_linear(log_space_mask, linear_spec_shape):
    # Always use CPU for debug visualization
    target_device = torch.device('cpu')
    log_space_mask = log_space_mask.to(target_device)
    linear_height, linear_width = linear_spec_shape
    
    y = torch.linspace(-1, 1, linear_height, device=target_device)
    x = torch.linspace(-1, 1, linear_width, device=target_device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    y_norm = (grid_y + 1) / 2
    log_y_norm = torch.log10(y_norm * 9 + 1)
    log_grid_y = log_y_norm * 2 - 1

    target_grid = torch.stack((grid_x, log_grid_y), dim=-1).unsqueeze(0)
    log_space_mask_unsqueezed = log_space_mask.float().unsqueeze(0).unsqueeze(0)
    
    linear_mask = F.grid_sample(
        log_space_mask_unsqueezed, target_grid, 
        mode='bilinear', padding_mode='border', align_corners=False
    )
    return linear_mask.squeeze(0).squeeze(0)

def extract_event_audio(grid_data_loc, start_time: datetime, end_time: datetime, sample_rate: int) -> torch.Tensor:
    duration_s = (end_time - start_time).total_seconds()
    num_samples = int(duration_s * sample_rate)
    output_waveform = torch.zeros(1, num_samples)
    overlapping_files = [f for f in grid_data_loc if f['start_time'] < end_time and f['end_time'] > start_time]
    
    for audio_file in overlapping_files:
        try:
            intersect_start = max(start_time, audio_file['start_time'])
            intersect_end = min(end_time, audio_file['end_time'])
            if intersect_end <= intersect_start: continue

            # Calc offsets
            file_off_start = (intersect_start - audio_file['start_time']).total_seconds()
            req_dur = (intersect_end - intersect_start).total_seconds()
            
            # Load with Torchaudio
            frame_offset = int(file_off_start * sample_rate)
            num_frames = int(req_dur * sample_rate)
            
            fw, sr = torchaudio.load(audio_file['path'], frame_offset=frame_offset, num_frames=num_frames)
            if sr != sample_rate: fw = torchaudio.functional.resample(fw, sr, sample_rate)
            
            # Place in output
            out_off_start = (intersect_start - start_time).total_seconds()
            out_idx = int(out_off_start * sample_rate)
            end_idx = out_idx + fw.shape[1]
            if end_idx > num_samples: 
                fw = fw[:, :num_samples-out_idx]
                end_idx = num_samples
            
            output_waveform[:, out_idx:end_idx] = fw
        except Exception as e:
            print(f"Error loading {os.path.basename(audio_file['path'])}: {e}")
    return output_waveform

def reconstruct_audio_for_localization(full_clip, det, event_start_time, stft_params, device):
    """
    Reconstructs isolated audio using the exact mask.
    """
    # 1. Compute Complex Spectrogram of the Full Event
    transform = torchaudio.transforms.Spectrogram(
        n_fft=stft_params['n_fft'], 
        win_length=stft_params['win_length'], 
        hop_length=stft_params['hop_length'], 
        power=None
    ).to(device)
    
    clip_t = full_clip.to(device)
    if clip_t.dim() == 1: clip_t = clip_t.unsqueeze(0)
    
    complex_spec = transform(clip_t).squeeze(0) # (Freq_Linear, Time_Frames)
    
    # 2. Prepare the Canvas Mask (Same size as spectrogram, initialized to zeros)
    canvas_mask = torch.zeros(complex_spec.shape, device=device)
    
    # 3. Process the Detection Mask
    mask_np = det['mask']
    mask_t = torch.from_numpy(mask_np).float().to(device)
    mask_t = torch.flip(mask_t, dims=[0]) # Flip vertically
    
    # Calculate Target Dimensions for the Slice
    target_freq_bins = complex_spec.shape[0]
    target_width_pixels = mask_np.shape[1]
    
    # Resample ONLY the slice
    linear_mask_slice = resample_log_mask_to_linear(mask_t, (target_freq_bins, target_width_pixels))
    
    # Thresholding
    upscaled_mask_slice = (linear_mask_slice > 0.5).float()
    
    # 4. Calculate Insertion Position
    sample_rate = 48000 
    seconds_per_frame = stft_params['hop_length'] / sample_rate 
    
    # Start time of the specific detection vs Start time of the whole event clip
    time_offset_s = (det['absolute_start'] - event_start_time).total_seconds()
    
    start_frame = int(time_offset_s / seconds_per_frame)
    end_frame = start_frame + upscaled_mask_slice.shape[1]
    
    # Bounds checking
    start_frame = max(0, start_frame)
    end_frame = min(canvas_mask.shape[1], end_frame)
    
    # Determine width to paste (handle edge clipping)
    paste_width = end_frame - start_frame
    
    if paste_width > 0:
        canvas_mask[:, start_frame:end_frame] = upscaled_mask_slice[:, :paste_width]
    
    # 5. Apply Mask & ISTFT
    masked_spec = complex_spec * canvas_mask
    
    istft = torchaudio.transforms.InverseSpectrogram(
        n_fft=stft_params['n_fft'], 
        win_length=stft_params['win_length'], 
        hop_length=stft_params['hop_length']
    ).to(device)
    
    isolated_audio = istft(masked_spec.unsqueeze(0)).squeeze(0)
    return isolated_audio.cpu()

def compute_pairwise_lag(sig1_tensor, sig2_tensor, valid_range1, valid_range2, sample_rate, max_tdoa_s):
    """
    Computes lag of sig2 relative to sig1.
    """
    max_lag_samples = int(np.ceil(max_tdoa_s * sample_rate))
    inter_start = max(valid_range1[0], valid_range2[0])
    inter_end = min(valid_range1[1], valid_range2[1])
    
    if inter_end <= inter_start:
        return None, 0.0, None, None
    
    s1 = sig1_tensor.numpy()[inter_start:inter_end]
    s2 = sig2_tensor.numpy()[inter_start:inter_end]

    # Normalization (Max-Abs)
    s1 = s1 / (np.max(np.abs(s1)) + 1e-9)
    s2 = s2 / (np.max(np.abs(s2)) + 1e-9)

    # Pad ONLY s2
    pad_width = (max_lag_samples, max_lag_samples)
    s2_padded = np.pad(s2, pad_width, mode='constant', constant_values=0)
    
    # Correlate
    corr = correlate(s2_padded, s1, mode='valid', method='fft')
 
    lags_axis_samples = np.arange(len(corr)) - max_lag_samples
    lags_seconds_axis = lags_axis_samples / sample_rate

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]
    
    calculated_lag = lags_seconds_axis[peak_idx]

    return calculated_lag, peak_val, corr, lags_seconds_axis

def initialize_grid(grid_dir, mapping_file, sample_rate):
    print("Initializing Audio Grid...")
    with open(mapping_file, 'r') as f: mic_to_loc_id = json.load(f)
    files = glob.glob(os.path.join(grid_dir, '**', '*.wav'), recursive=True) + glob.glob(os.path.join(grid_dir, '**', '*.WAV'), recursive=True)
    
    grid = defaultdict(list)
    for path in files:
        meta = extract_guano_metadata_with_duration(path, sample_rate)
        if not meta: continue
        try:
            ts = datetime.fromisoformat(meta['Timestamp'].replace('Z', '+00:00'))
            # Apply manual offset only (ignoring sync offset for raw debug view)
            did = meta['Serial']
            manual_offset = mic_to_loc_id.get(did, {}).get('initial_offset_ms', 0)
            start_time = ts - timedelta(milliseconds=manual_offset)
            end_time = start_time + timedelta(seconds=meta['duration_s'])
            
            lid = get_location_id(did, ts, mic_to_loc_id)
            if lid:
                grid[lid].append({'path': path, 'start_time': start_time, 'end_time': end_time})
        except Exception: continue
    
    for k in grid: grid[k].sort(key=lambda x: x['start_time'])
    print(f"Grid Initialized. Found data for {len(grid)} locations.")
    return grid

def load_sensor_locations(csv_path: str) -> dict:
    """Loads lat/long/elevation from CSV and converts to local Cartesian (meters) [x, y, z]."""
    df = pd.read_csv(csv_path)
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    center_ele = df['elevation'].mean()
    center_point = (center_lat, center_lon)
    
    local_coords = {}
    for _, row in df.iterrows():
        dist_y = geopy_distance(center_point, (row['latitude'], center_lon)).meters
        if row['latitude'] < center_lat: dist_y = -dist_y
        dist_x = geopy_distance(center_point, (center_lat, row['longitude'])).meters
        if row['longitude'] < center_lon: dist_x = -dist_x
        
        # Calculate Z (elevation difference relative to array center)
        dist_z = row['elevation'] - center_ele
        
        local_coords[int(row['locationID'])] = np.array([dist_x, dist_y, dist_z])
    return local_coords

def get_ground_plane(mic_coords):
    """Returns (point_on_plane, normal_vector) for the plane defined by mics 1, 3, 8."""
    if not all(k in mic_coords for k in [1, 3, 8]):
        return None, None
        
    p1 = mic_coords[1]
    p3 = mic_coords[3]
    p8 = mic_coords[8]
    
    v1 = p3 - p1
    v2 = p8 - p1
    normal = np.cross(v1, v2)
    
    # Ensure normal points 'Up' (Positive Z)
    if normal[2] < 0: normal = -normal
    
    # Normalize
    normal = normal / np.linalg.norm(normal)
    
    return p1, normal

def apply_ground_constraint(grid_points, mic_coords):
    p1, normal = get_ground_plane(mic_coords)
    if p1 is None: return grid_points

    # Vector from P1 to all points
    vecs = grid_points - p1
    
    # Dot product with normal (signed distance)
    dists = np.dot(vecs, normal)
    
    # Keep points on or above plane (with small buffer for float errors)
    return grid_points[dists >= -0.5]

def define_search_grid_3d(mic_coords, grid_size, resolution=2.0, z_range=100):
    """
    Defines a 3D search grid centered on the microphone array.
    """
    coords = np.array(list(mic_coords.values()))
    center = coords.mean(axis=0)
    
    width, height = grid_size
    min_x, max_x = center[0] - width / 2, center[0] + width / 2
    min_y, max_y = center[1] - height / 2, center[1] + height / 2
    min_z, max_z = center[2] - z_range, center[2] + z_range
    
    x_range = np.arange(min_x, max_x, resolution)
    y_range = np.arange(min_y, max_y, resolution)
    z_range = np.arange(min_z, max_z, resolution)
    
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    
    # Apply Ground Plane Constraint (Mics 1, 3, 8)
    grid_points = apply_ground_constraint(grid_points, mic_coords)
    
    return grid_x, grid_y, grid_z, grid_points

# Multi-resolution approach
def solve_location_multiresolution(mic_coords, tdoas, ref_idx):
    """
    Coarse 3D grid -> Medium 3D grid -> Fine continuous optimization
    """
    c = 343.0
    ref_pos = mic_coords[ref_idx]
    
    # Phase 1: Coarse 3D grid (10m resolution, ±100m vertical)
    _, _, _, grid_coarse = define_search_grid_3d(
        mic_coords, (800, 800), resolution=10.0, z_range=100
    )
    
    total_error = compute_grid_errors(mic_coords, tdoas, ref_idx, grid_coarse, c)
    coarse_best_idx = np.argmin(total_error)
    coarse_pos = grid_coarse[coarse_best_idx]
    
    # Phase 2: Fine 3D grid around coarse estimate (2m resolution, ±50m cube)
    fine_x = np.arange(coarse_pos[0] - 50, coarse_pos[0] + 50, 2.0)
    fine_y = np.arange(coarse_pos[1] - 50, coarse_pos[1] + 50, 2.0)
    fine_z = np.arange(coarse_pos[2] - 50, coarse_pos[2] + 50, 2.0)
    grid_fx, grid_fy, grid_fz = np.meshgrid(fine_x, fine_y, fine_z)
    grid_fine = np.vstack([grid_fx.ravel(), grid_fy.ravel(), grid_fz.ravel()]).T
    
    # Apply constraint to fine grid as well
    grid_fine = apply_ground_constraint(grid_fine, mic_coords)

    if len(grid_fine) == 0:
        # Fallback if fine grid is entirely clipped (unlikely if coarse found something valid)
        grid_fine = np.array([coarse_pos]) 
        print(f'Warning: Fine grid was fully clipped by ground constraint. Using coarse position only.')
    
    total_error_fine = compute_grid_errors(mic_coords, tdoas, ref_idx, grid_fine, c)
    fine_best_idx = np.argmin(total_error_fine)
    fine_pos = grid_fine[fine_best_idx]
    
    # Get Plane info for Continuous Optimization Constraint
    plane_p1, plane_normal = get_ground_plane(mic_coords)

    # Phase 3: Continuous refinement
    def objective_func(pos):
        err = 0.0
        d_ref = np.linalg.norm(pos - ref_pos)
        for loc_id, t_meas in tdoas.items():
            if loc_id == ref_idx: continue
            d_mic = np.linalg.norm(pos - mic_coords[loc_id])
            err += ((d_mic - d_ref) - t_meas * c) ** 2
            
        # --- Soft Constraint: Ground Plane Penalty ---
        if plane_p1 is not None:
            # signed distance to plane
            dist = np.dot(pos - plane_p1, plane_normal)
            if dist < 0:
                # Heavy penalty for being underground: 
                # Square the violation distance and multiply by large factor
                err += 1e6 * (dist ** 2)
        
        return err
    
    bounds = [
        (fine_pos[0] - 100, fine_pos[0] + 100),
        (fine_pos[1] - 100, fine_pos[1] + 100),
        (fine_pos[2] - 100, fine_pos[2] + 100)
    ]
    
    res = minimize(objective_func, fine_pos, method='L-BFGS-B', bounds=bounds)
    
    return res.x if res.success else fine_pos, res.fun if res.success else total_error_fine[fine_best_idx]

def compute_grid_errors(mic_coords, tdoas, ref_idx, grid_points, c):
    """Vectorized error computation for grid search"""
    ref_pos = mic_coords[ref_idx]
    total_error = np.zeros(len(grid_points))
    
    for loc_id, tdoa_measured in tdoas.items():
        if loc_id == ref_idx: continue
        
        mic_pos = mic_coords[loc_id]
        dist_to_ref = np.linalg.norm(grid_points - ref_pos, axis=1)
        dist_to_mic = np.linalg.norm(grid_points - mic_pos, axis=1)
        
        ddoa_theoretical = dist_to_mic - dist_to_ref
        ddoa_measured = tdoa_measured * c
        
        total_error += (ddoa_theoretical - ddoa_measured) ** 2
    
    return total_error

# --- Optimization Logic ---

def _global_optimization_cost(x, optimizable_mics, ref_mic, base_offsets, simplified_groups, mic_coords, grid_points):
    """
    Cost function for the optimizer. Calculates total residual error for offset candidates.
    Must be top-level for multiprocessing pickling.
    """
    # Reconstruct offsets from vector x
    temp_offsets = base_offsets.copy()
    for i, m in enumerate(optimizable_mics):
        temp_offsets[m] = x[i]
    
    total_residual_error = 0.0
    
    # Iterate over groups
    for mics, lag_map in simplified_groups:
        if len(mics) < 2: continue
        
        local_ref = mics[0]
        tdoas = {}
        valid_group = True
        
        for m_target in mics:
            if m_target == local_ref: continue
            
            # Key lookup: tuple sorted by ID
            key = tuple(sorted((local_ref, m_target)))
            if key not in lag_map:
                valid_group = False
                break
                
            # If key is (ref, target), lag is target - ref. 
            # If key is (target, ref), lag is ref - target (so we negate).
            raw_lag = lag_map[key]
            measured_lag = raw_lag if key == (local_ref, m_target) else -raw_lag
            
            # Apply offset correction: (T_meas) - (Offset_target - Offset_ref)
            adj_lag = measured_lag - (temp_offsets[m_target] - temp_offsets[local_ref])
            tdoas[m_target] = adj_lag
        
        if not valid_group: continue

        # Calculate error on the grid
        est_pos,  min_err = solve_location_multiresolution(
            mic_coords, tdoas, local_ref
        )
        total_residual_error += min_err
        
    return total_residual_error

class BlindSynchroniser:
    def __init__(self, processed_groups, mic_coords, active_mics):
        self.groups = processed_groups
        self.mic_coords = mic_coords
        self.active_mics = sorted(list(active_mics))
        self.offsets = {m: 0.0 for m in self.active_mics}
        
        # Pre-calc Grid for localization steps
        # 800x800m grid
        self.grid_x, self.grid_y, self.grid_z, self.grid_points = define_search_grid_3d(mic_coords, (800, 800), resolution=10.0, z_range=100)
        
    def get_pairwise_lag(self, m_a, m_b, pairwise_data):
        key = tuple(sorted((m_a, m_b)))
        if key not in pairwise_data: return None
        raw_lag = pairwise_data[key][0] # (lag, peak_corr, ...)
        return raw_lag if m_a < m_b else -raw_lag

    def get_consistent_edges(self, group):
        """Returns set of edges forming valid cycles within tolerance."""
        active = group['active_mics']
        pairwise = group['pairwise_data']
        valid = set()
        
        for m1, m2, m3 in combinations(active, 3):
            l12 = self.get_pairwise_lag(m1, m2, pairwise)
            l23 = self.get_pairwise_lag(m2, m3, pairwise)
            l13 = self.get_pairwise_lag(m1, m3, pairwise)
            
            if l12 is not None and l23 is not None and l13 is not None:
                # Cycle check: t12 + t23 ≈ t13
                if abs((l12 + l23) - l13) < CONSISTENCY_THRESHOLD_S:
                    valid.add(tuple(sorted((m1, m2))))
                    valid.add(tuple(sorted((m2, m3))))
                    valid.add(tuple(sorted((m1, m3))))
        return valid

    def run(self):
        print(f"\n> Starting Blind Synchronization on {len(self.active_mics)} mics using {len(self.groups)} groups.")
        
        # 1. Select Reference Mic
        ref_mic = self.active_mics[0]
        optimizable_mics = [m for m in self.active_mics if m != ref_mic]
        print(f"> Reference: Mic {ref_mic}. Optimizing: {optimizable_mics}")

        # 2. Filter Groups & Prepare Data
        simplified_groups = []
        pair_count = 0
        
        for g in self.groups:
            valid_edges = self.get_consistent_edges(g)
            if not valid_edges: continue
            
            lag_map = {}
            for k, v in g['pairwise_data'].items():
                if k in valid_edges:
                    lag_map[k] = v[0] # Just the lag value
                    pair_count += 1
            
            if lag_map:
                simplified_groups.append((g['active_mics'], lag_map))

        print(f"> Prepared {len(simplified_groups)} consistent groups ({pair_count} pairs) for optimization.")
        
        if not simplified_groups:
            print("> No consistent groups found. Aborting.")
            return

        # 3. Global Search (Differential Evolution)
        # Coarse grid for speed
        coarse_grid = self.grid_points[::4]
        bounds = [(-0.5, 0.5) for _ in optimizable_mics] # +/- 0.5s search window
        base_offsets = self.offsets.copy()
        
        print("> Phase 1: Differential Evolution (Global Search)...")
        res_global = differential_evolution(
            _global_optimization_cost,
            bounds,
            args=(optimizable_mics, ref_mic, base_offsets, simplified_groups, self.mic_coords, coarse_grid),
            strategy='best2bin',
            maxiter=100,
            popsize=50,
            tol=0.1,
            workers=-1,
            disp=True
        )
        print(f"> Global Cost: {res_global.fun:.4f}")

        # print offsets after global
        print("> Offsets after Global Search:")
        # temporarily shift to print relative to ref
        temp_offsets = base_offsets.copy()
        for i, m in enumerate(optimizable_mics):
            temp_offsets[m] = res_global.x[i]
        shift = temp_offsets[ref_mic]
        for m in temp_offsets:
            temp_offsets[m] -= shift
            print(f"  Mic {m}: {temp_offsets[m]*1000:+.2f} ms")

        # 4. Local Polish (Nelder-Mead)
        print("> Phase 2: Nelder-Mead (Fine Polish)...")
        res_local = minimize(
            _global_optimization_cost, 
            res_global.x,
            args=(optimizable_mics, ref_mic, base_offsets, simplified_groups, self.mic_coords, self.grid_points),
            method='Nelder-Mead', 
            options={'maxiter': 800, 'xatol': 1e-4, 'fatol': 1e-4}
        )
        
        # 5. Apply Results
        final_vals = res_local.x
        for i, m in enumerate(optimizable_mics):
            self.offsets[m] = final_vals[i]
        
        # Normalize (Shift so Ref=0)
        shift = self.offsets[ref_mic]
        for m in self.offsets:
            self.offsets[m] -= shift

        print("\n> Optimization Complete.")
        print("> Final Offsets (relative to Ref):")
        for m in self.active_mics:
            print(f"  Mic {m}: {self.offsets[m]*1000:+.2f} ms")

        return self.offsets

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def generate_rainbow_colors(num_colors: int) -> list[str]:
    if not isinstance(num_colors, int) or num_colors <= 0: return []
    hex_colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb_float = colorsys.hls_to_rgb(hue, 0.5, 1.0)
        rgb_int = tuple(int(c * 255) for c in rgb_float)
        hex_colors.append(f"#{rgb_int[0]:02x}{rgb_int[1]:02x}{rgb_int[2]:02x}".upper())
    return hex_colors

# --- Volumetric Helper ---
def generate_cloud_data(center, tdoas, ref_mic, mic_coords):
    """Generates grid points and likelihoods around a center estimate."""
    c = 343.0
    
    box_radius = 100.0
    res = 6.0
    
    # Define local grid
    x = np.arange(center[0]-box_radius, center[0]+box_radius, res)
    y = np.arange(center[1]-box_radius, center[1]+box_radius, res)
    z = np.arange(center[2]-box_radius, center[2]+box_radius, res)
    
    gx, gy, gz = np.meshgrid(x, y, z)
    grid_points = np.vstack([gx.ravel(), gy.ravel(), gz.ravel()]).T
    
    # Apply Ground Constraint to Visualization Cloud
    grid_points = apply_ground_constraint(grid_points, mic_coords)

    if len(grid_points) == 0: return np.array([]), np.array([])
    
    # Compute Error
    ref_pos = mic_coords[ref_mic]
    
    # Initialize with 0
    total_error = np.zeros(len(grid_points))
    
    dist_to_ref = np.linalg.norm(grid_points - ref_pos, axis=1)
    
    active_mic_count = 0
    for loc_id, tdoa_val in tdoas.items():
        mic_pos = mic_coords[loc_id]
        dist_to_mic = np.linalg.norm(grid_points - mic_pos, axis=1)
        ddoa_theoretical = dist_to_mic - dist_to_ref
        ddoa_measured = tdoa_val * c
        
        # Sum of squared errors
        total_error += (ddoa_theoretical - ddoa_measured) ** 2
        active_mic_count += 1

    if active_mic_count == 0: return np.array([]), np.array([])

    min_err = np.min(total_error)
    
    sigma = 50 
    likelihood = np.exp(-(total_error - min_err) / sigma)
    mask = likelihood > 0.02
    
    return grid_points[mask], likelihood[mask]

def visualize_geometry_3d(mic_coords, estimated_positions, title_context, active_offsets):
    """
    Plots sensor array and localized events with a time slider.
    Renders a softer, larger volumetric likelihood cloud around estimates.
    """
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15)
    
    # 1. Plot microphones (static)
    xs = [coords[0] for coords in mic_coords.values()]
    ys = [coords[1] for coords in mic_coords.values()]
    zs = [coords[2] for coords in mic_coords.values()]
    
    ax.scatter(xs, ys, zs, c='red', marker='^', s=50, label='Mics', depthshade=False, edgecolors='k')
    for mid, coords in mic_coords.items():
        label = f"m{mid}"
        if mid in active_offsets:
            label += f"\n{active_offsets[mid]*1000:.1f}ms"
        ax.text(coords[0], coords[1], coords[2], label, color='black', fontsize=8)

    if not estimated_positions:
        plt.show()
        return

    # Sort and Normalize Time
    est_sorted = sorted(estimated_positions, key=lambda x: x['time'])
    start_time = est_sorted[0]['time']
    end_time = est_sorted[-1]['time']
    total_duration = (end_time - start_time).total_seconds()
    
    for p in est_sorted:
        p['rel_time'] = (p['time'] - start_time).total_seconds()

    # Pre-calculate colors
    unique_species = sorted(list(set(p['species'] for p in est_sorted)))
    hex_colors = generate_rainbow_colors(len(unique_species))
    species_color_map = {sp: [x / 255.0 for x in hex_to_rgb(col)] for sp, col in zip(unique_species, hex_colors)}

    # Create Legend
    legend_handles = []
    for sp, color in species_color_map.items():
        # Create a dummy point for the legend entry
        h = ax.scatter([], [], c=[color], marker='*', s=100, label=sp, edgecolors='white')
        legend_handles.append(h)
    ax.legend(handles=legend_handles, loc='upper left', title="Species", fontsize='small')

    active_scatters = []
    
    # --- Update Function ---
    time_window = 4.0 
    
    def update_plot(val):
        for scat in active_scatters: scat.remove()
        active_scatters.clear()
        
        current_t = slider_time.val
        visible = [p for p in est_sorted if current_t <= p['rel_time'] < current_t + time_window]
        
        for p in visible:
            base_color = species_color_map[p['species']]
            
            # 1. Plot the "Best Guess" Star
            s1 = ax.scatter(
                p['pos'][0], p['pos'][1], p['pos'][2],
                color=base_color, marker='*', s=150, edgecolors='white', zorder=10
            )
            active_scatters.append(s1)
            
            # 2. Generate and Plot the Likelihood Cloud
            cloud_pts, cloud_lik = generate_cloud_data(p['pos'], p['tdoas'], p['ref_mic'], mic_coords)
            
            if len(cloud_pts) > 0:
                # Create an RGBA array
                colors = np.zeros((len(cloud_pts), 4))
                colors[:, 0:3] = base_color
                
                # Soften the alpha: 
                # Multiply likelihood by 0.15 for maximum transparency (faint cloud)
                # This ensures even the "peak" isn't a solid block, allowing us to see structure
                colors[:, 3] = cloud_lik * 0.15 
                
                s2 = ax.scatter(
                    cloud_pts[:, 0], cloud_pts[:, 1], cloud_pts[:, 2],
                    c=colors, marker='s', s=25, depthshade=False, zorder=5, linewidths=0
                )
                active_scatters.append(s2)

        time_str = (start_time + timedelta(seconds=current_t)).strftime('%H:%M:%S')
        ax.set_title(f"{title_context}\nTime: {time_str} | Visible Events: {len(visible)}")
        fig.canvas.draw_idle()

    # Controls
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
    slider_time = Slider(ax_slider, 'Time (s)', 0, total_duration, valinit=0, valstep=0.1)
    slider_time.on_changed(update_plot)
    
    ax_play = plt.axes([0.85, 0.05, 0.1, 0.04])
    btn_play = Button(ax_play, 'Play')
    
    class Player:
        def __init__(self):
            self.playing = False
            self.anim = None
        def toggle(self, event):
            self.playing = not self.playing
            if self.playing:
                self.anim = animation.FuncAnimation(fig, self.step, frames=None, interval=50, cache_frame_data=False)
                plt.draw()
            else:
                if self.anim: self.anim.event_source.stop()
        def step(self, frame):
            if not self.playing: return
            val = slider_time.val + 0.1
            if val > total_duration: val = 0
            slider_time.set_val(val)

    player = Player()
    btn_play.on_clicked(player.toggle)

    # View Limits based on estimates
    all_pts = np.array([p['pos'] for p in est_sorted])
    mid = np.mean(all_pts, axis=0)
    max_range = np.max(np.abs(all_pts - mid)) + 100 # Add buffer for clouds
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(np.min(zs)-20, np.max(zs)+100) 
    
    update_plot(0)
    plt.show()

def save_localisation_results(estimated_positions, mic_coords, output_dir):
    """
    Saves 3D likelihood volumes and metadata to disk in JSON format.
    Each detection gets its own file with grid points and likelihood values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata about the array
    array_metadata = {
        "microphones": {
            str(mic_id): {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2])
            }
            for mic_id, coords in mic_coords.items()
        },
        "num_detections": len(estimated_positions),
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = os.path.join(output_dir, "array_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(array_metadata, f, indent=2)
    print(f"Saved array metadata to {metadata_path}")
    
    # Save each detection's likelihood volume
    for idx, detection in enumerate(estimated_positions):
        # Generate likelihood cloud data
        cloud_pts, cloud_lik = generate_cloud_data(
            detection['pos'], 
            detection['tdoas'], 
            detection['ref_mic'], 
            mic_coords
        )
        
        if len(cloud_pts) == 0:
            continue
        
        # Create detection record
        detection_data = {
            "detection_id": idx,
            "species": detection['species'],
            "timestamp": detection['time'].isoformat(),
            "mic_count": detection['mic_count'],
            "ref_mic": detection['ref_mic'],
            "best_estimate": {
                "x": float(detection['pos'][0]),
                "y": float(detection['pos'][1]),
                "z": float(detection['pos'][2])
            },
            "likelihood_cloud": {
                "points": [
                    {
                        "x": float(pt[0]),
                        "y": float(pt[1]),
                        "z": float(pt[2]),
                        "likelihood": float(lik)
                    }
                    for pt, lik in zip(cloud_pts, cloud_lik)
                ]
            }
        }
        
        # Save to individual file
        filename = f"detection_{idx:04d}_{detection['time'].strftime('%H%M%S')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        if (idx + 1) % 10 == 0:
            print(f"  Saved {idx + 1}/{len(estimated_positions)} detections...")
    
    print(f"\nSaved {len(estimated_positions)} likelihood volumes to {output_dir}/")

# --- Main Logic ---

def main():
    # 1. Setup
    print("--- Blind Synchronisation Tool ---")
    if not os.path.exists(PICKLE_DIR):
        print(f"Error: Pickle directory '{PICKLE_DIR}' not found.")
        return

    mic_coords = load_sensor_locations(LOCATION_FILE)
    grid_data = initialize_grid(GRID_AUDIO_DIR, MAPPING_FILE, SAMPLE_RATE)
    
    pickle_paths = glob.glob(os.path.join(PICKLE_DIR, "*.pkl"))
    if not pickle_paths:
        print("No .pkl files found.")
        return
    
    # sort by group index
    pickle_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    print(f"> Found {len(pickle_paths)} group files. Processing audio to extract lags...")
    
    processed_groups_pool = []
    all_active_mics = set()

    # 2. Process Files
    # We process ALL available files first to build a pool of valid pairwise lags
    for i, path in enumerate(pickle_paths):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            group_id = data.get('group_id', 'Unknown')
            raw_species = data.get('species_name', 'Unknown')
            species_name = SPECIES_DISPLAY_NAMES.get(raw_species, raw_species)
            dets = data['detections']
            
            # Determine global window
            all_starts = [d['absolute_start'] for d in dets]
            all_ends = [d['absolute_end'] for d in dets]
            g_start = min(all_starts) - timedelta(seconds=0.2)
            g_end = max(all_ends) + timedelta(seconds=0.2)
            
            # Fetch Raw Audio
            raw_signals = {}
            for loc_id in range(1, 10): # Assuming IDs 1-9
                files = grid_data.get(loc_id, [])
                if files:
                    wav = extract_event_audio(files, g_start, g_end, SAMPLE_RATE)
                    if wav.abs().sum() > 0:
                        raw_signals[loc_id] = wav.squeeze()
            
            # Reconstruct Isolated Audio
            core_signals = {}
            valid_windows = {}
            
            for d in dets:
                lid = d['loc_id']
                if lid not in raw_signals: continue
                
                # Reconstruct
                full = raw_signals[lid]
                iso = reconstruct_audio_for_localization(full, d, g_start, STFT_PARAMS, torch.device('cpu'))
                
                # Length fix
                if iso.shape[0] != full.shape[0]:
                    target = full.shape[0]
                    iso = iso[:target] if iso.shape[0] > target else torch.cat([iso, torch.zeros(target - iso.shape[0])])
                
                core_signals[lid] = iso
                valid_windows[lid] = (0, full.shape[0])

            if len(core_signals) < 3: continue
            
            # Compute Lags
            mics = sorted(list(core_signals.keys()))
            pairwise_data = {}
            
            for m1, m2 in permutations(mics, 2):
                lag, peak, corr, _ = compute_pairwise_lag(
                    core_signals[m1], core_signals[m2],
                    valid_windows[m1], valid_windows[m2],
                    SAMPLE_RATE, max_tdoa_s=4.0
                )
                if lag is not None:
                    pairwise_data[(m1, m2)] = (lag, peak) # minimal data needed
            
            processed_groups_pool.append({
                'id': group_id,
                'species': species_name,
                'active_mics': mics,
                'pairwise_data': pairwise_data,
                'timestamp': g_start
            })
            all_active_mics.update(mics)
            
            if (i+1) % 10 == 0:
                print(f"  > Processed {i+1}/{len(pickle_paths)}...")

        except Exception as e:
            print(f"  > Error processing {os.path.basename(path)}: {e}")

    # 3. Run Sync Bootstrap Loop OR Load Existing
    final_averaged_offsets = {}

    if SKIP_SYNCHRONISATION:
        print(f"\n> SKIP_SYNCHRONISATION=True. Loading offsets from '{OFFSETS_FILE}'...")
        if not os.path.exists(OFFSETS_FILE):
            print(f"Error: {OFFSETS_FILE} not found. Cannot skip synchronisation.")
            return
        
        try:
            with open(OFFSETS_FILE, 'r') as f:
                loaded_data = json.load(f)
                # JSON keys are strings, convert to int for loc_ids
                final_averaged_offsets = {int(k): v for k, v in loaded_data.items()}
            print("> Offsets loaded successfully.")
        except Exception as e:
            print(f"Error loading offsets: {e}")
            return

    else:
        if len(processed_groups_pool) < GROUPS_PER_ROUND:
            print(f"Error: Not enough valid groups found ({len(processed_groups_pool)}) to sample {GROUPS_PER_ROUND}.")
            return

        print(f"\n> Starting Bootstrap Analysis ({NUM_BOOTSTRAP_ROUNDS} rounds, {GROUPS_PER_ROUND} groups per round)...")
        
        results_by_mic = defaultdict(list)
        
        for round_idx in range(NUM_BOOTSTRAP_ROUNDS):
            print(f"\n--- Round {round_idx + 1}/{NUM_BOOTSTRAP_ROUNDS} ---")
            
            # Randomly sample groups for this iteration
            current_sample = random.sample(processed_groups_pool, GROUPS_PER_ROUND)
            
            sync_tool = BlindSynchroniser(current_sample, mic_coords, all_active_mics)
            
            try:
                round_offsets = sync_tool.run()
                if round_offsets:
                    for mic, offset in round_offsets.items():
                        results_by_mic[mic].append(offset)
            except Exception as e:
                print(f"Round {round_idx+1} failed: {e}")
                import traceback
                traceback.print_exc()

        # 4. Calculate Statistics
        print("\n\n==========================================")
        print("       BOOTSTRAP STABILITY RESULTS        ")
        print("==========================================")
        print(f"{'Mic':<5} | {'Max Abs Diff (ms)':<20} | {'Std Dev (ms)':<15} | {'Mean Offset (ms)':<15}")
        print("-" * 65)

        sorted_mics = sorted(results_by_mic.keys())
        for mic in sorted_mics:
            offsets = np.array(results_by_mic[mic])
            
            if len(offsets) < 2:
                print(f"{mic:<5} | {'Insufficient Data':<20}")
                continue

            offsets_ms = offsets * 1000.0
            max_diff = np.max(offsets_ms) - np.min(offsets_ms)
            std_dev = np.std(offsets_ms)
            mean_val = np.mean(offsets_ms)
            final_averaged_offsets[mic] = mean_val / 1000.0 # Store back in seconds
            
            print(f"{mic:<5} | {max_diff:20.4f} | {std_dev:15.4f} | {mean_val:15.4f}")

        with open(OFFSETS_FILE, 'w') as f:
            json.dump({str(k): v for k, v in final_averaged_offsets.items()}, f, indent=4)
        print(f"\n> Saved averaged offsets to {OFFSETS_FILE}")

    # 5. Final Visualization (All Groups)
    print(f"\n> Visualizing all {len(processed_groups_pool)} groups using final offsets...")
    
    final_estimates = []

    for group in processed_groups_pool:
        mics = sorted(group['active_mics'])
        if len(mics) < 3: continue
        
        ref = mics[0]
        tdoas = {}
        valid_group_calc = True
        
        for m in mics:
            if m == ref: continue
            
            if m not in final_averaged_offsets or ref not in final_averaged_offsets:
                valid_group_calc = False
                break

            key = tuple(sorted((ref, m)))
            if key not in group['pairwise_data']:
                valid_group_calc = False
                break
                
            raw_lag = group['pairwise_data'][key][0]
            measured_lag = raw_lag if ref < m else -raw_lag
            
            # Apply the FINAL averaged offsets
            correction = final_averaged_offsets[m] - final_averaged_offsets[ref]
            tdoas[m] = measured_lag - correction
        
        if valid_group_calc:
            est_pos, _ = solve_location_multiresolution(
                mic_coords, tdoas, ref
            )
            final_estimates.append({
                'pos': est_pos, 
                'species': group['species'], 
                'mic_count': len(mics),
                'time': group['timestamp'],
                'tdoas': tdoas,
                'ref_mic': ref
            })

    if SAVE_LOCALISATION_DATA and final_estimates:
        print(f"\n> Saving localisation results to '{LOCALISATION_OUTPUT_DIR}'...")
        save_localisation_results(final_estimates, mic_coords, LOCALISATION_OUTPUT_DIR)


    visualize_geometry_3d(mic_coords, final_estimates, "All Groups (Final Offsets)", final_averaged_offsets)

if __name__ == "__main__":
    main()