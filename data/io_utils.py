import os
import numpy as np
import scipy.io as sio
import scipy.signal
import re
import pickle  # <--- 必须显式导入这个！

def get_load_data_func(dataset_name):
    if dataset_name == 'SEEDV':
        return load_processed_SEEDV_NEW_data
    elif dataset_name == 'SEED':
        return load_processed_SEED_NEW_data
    elif dataset_name == 'FACED':
        return load_processed_FACED_NEW_data
    else:
        raise ValueError('dataset_name not found')

def load_EEG_data(data_dir, cfg):
    load_data_func = get_load_data_func(cfg.dataset_name)
    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_data_func(
                                data_dir, cfg.fs, cfg.n_channs, cfg.timeLen, cfg.timeStep, 
                                cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class)
    return data, onesub_labels, n_samples_onesub, n_samples_sessions

def load_finetune_EEG_data(data_dir, cfg):
    load_data_func = get_load_data_func(cfg.dataset_name)
    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_data_func(
                                data_dir, cfg.fs, cfg.n_channs, cfg.timeLen2, cfg.timeStep2, 
                                cfg.n_session, cfg.n_subs, cfg.n_vids, cfg.n_class)
    return data, onesub_labels, n_samples_onesub, n_samples_sessions


def load_processed_FACED_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=1,
                                  n_subs=123, n_vids=28, n_class=9, t=30):
    """
    一步到位加载函数：
    1. 直接读取官方 .pkl (250Hz)
    2. 内存中降采样至 125Hz (fs参数)
    3. 执行 Z-score 标准化
    4. 执行滑动窗口切片
    """
    print(f"[One-Step Load] Reading FACED .pkl from: {dir}")
    print(f"Resampling: 250 Hz -> {fs} Hz | Window: {timeLen}s")

    list_files = os.listdir(dir)
    # 确保文件按 sub000, sub001... 排序，否则标签会乱
    # [新增] 过滤掉不相关的文件和文件夹，只保留 .pkl 文件
    list_files = [f for f in list_files if f.endswith('.pkl')]
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))

    # 官方数据原始采样率
    ORIG_FS = 250

    # 计算切片参数 (目标 fs=125)
    points_len = int(timeLen * fs)  # 5 * 125 = 625
    points_step = int(timeStep * fs)  # 2.5 * 125 = 312

    # 计算降采样后的总长度 (30秒 * 125Hz = 3750点)
    total_len_resampled = int(t * fs)

    # 计算切片数量
    n_samples = int((total_len_resampled - points_len) / points_step) + 1

    # 处理二分类/九分类的视频选择
    if n_class == 2:
        vid_sel = list(range(12)) + list(range(16, 28))
        n_vids_sel = 24
    elif n_class == 9:
        vid_sel = list(range(28))
        n_vids_sel = 28

    # 初始化大容器: (人数, 总样本数, 通道, 时间点)
    # n_subs=123, total_samples = 28 * n_samples
    total_samples_per_sub = n_vids_sel * n_samples
    data = np.zeros((n_subs, total_samples_per_sub, n_chans, points_len), dtype=np.float32)

    for idx, fn in enumerate(list_files):
        if idx >= n_subs: break

        file_path = os.path.join(dir, fn)

        # --- 1. 读取 .pkl ---
        try:
            with open(file_path, 'rb') as f:
                # 原始 shape: (28, 32, 7500)
                raw_user_data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {fn}: {e}")
            continue

        # --- 2. 降采样 (核心步骤) ---
        if fs != ORIG_FS:
            # 计算目标点数: 7500 * (125/250) = 3750
            target_points = int(raw_user_data.shape[2] * (fs / ORIG_FS))
            # axis=2 是时间维度
            user_data_resampled = scipy.signal.resample(raw_user_data, target_points, axis=2)
        else:
            user_data_resampled = raw_user_data

        # --- 3. Z-score 归一化 ---
        # 保持原作者逻辑：剔除极端异常值后计算均值方差
        thr = 30 * np.median(np.abs(user_data_resampled))
        mask = np.abs(user_data_resampled) < thr
        if mask.sum() > 0:
            mean_val = np.mean(user_data_resampled[mask])
            std_val = np.std(user_data_resampled[mask])
            user_data_norm = (user_data_resampled - mean_val) / (std_val + 1e-8)
        else:
            user_data_norm = user_data_resampled

        # --- 4. 切片填充 ---
        cnt = 0
        for vid in vid_sel:
            # 取出一个视频的数据 (32, 3750)
            vid_data = user_data_norm[vid]

            for i in range(n_samples):
                start = i * points_step
                end = start + points_len
                # 填入数据
                data[idx, cnt] = vid_data[:, start:end]
                cnt += 1

    # Reshape: (Total_Samples_All_Subs, Channels, Time)
    # 这步是为了符合后续 Dataset 的输入格式
    data = data.reshape(-1, n_chans, points_len)

    # --- 5. 标签生成 (匹配 Stimuli_info.xlsx) ---
    if n_class == 2:
        label_pattern = [0] * 12 + [1] * 12
    elif n_class == 9:
        # 严格对应 Excel: 3,3,3,3,4(Neutral),3,3,3,3
        label_pattern = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 4 + [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3

    onesub_labels = []
    for lbl in label_pattern:
        onesub_labels.extend([lbl] * n_samples)

    # 辅助变量，用于采样器
    n_samples_onesub = np.array([n_samples] * n_vids_sel)
    n_samples_sessions = n_samples_onesub.reshape(n_session, -1)

    print(f"Data loaded successfully. Output shape: {data.shape}")
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions


def load_processed_SEEDV_data(dir, fs, n_chans, timeLen,timeStep, n_session, n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (15*(sum(n_samples_onesub)))*62*point_len(1250)

    list_files = os.listdir(dir)
    list_files.sort(key= lambda x:int(x[:-4]))
    # print(list_files)

    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)


    n_samples_onesub = []
    for i in range(n_session):
        fn = list_files[i]
        file_path = os.path.join(dir,fn)
        onesubsession_data = sio.loadmat(file_path)  
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)

    n_samples_sum_onesub = np.sum(n_samples_onesub)


    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    s = np.arange(n_session)
    # n_samples_onesub = []
    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesubsession_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesubsession_data['data']   #(channels,tot_n_points)  (62,tot_n_points)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[EEG_data<thr])) / np.std(EEG_data[EEG_data<thr])
        n_points = np.squeeze(onesubsession_data['n_points']).astype(int)
        # print(EEG_data.shape)
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))
        n_samples_onesubsession = ((n_points-points_len)//points_step+1).astype(int)
        
        # if idx < n_session:
        #     if idx == s[idx]:
        #         n_samples_onesub = n_samples_onesub + list(n_samples_onesubsession)
        for vid in range(n_vids):
            # print('vid:',vid)
            for i in range(n_samples_onesubsession[vid]):
                # print('sample:',i)

                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1

                # 拼接速度会越来越慢
                # temp = temp.reshape(1,temp.shape[0],temp.shape[1])
                # start_time = time.time()
                # data = np.concatenate((data,temp),0)
                # end_time = time.time()
                # print(end_time - start_time)
    # print(cnt)

    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]
    
    print('load processed data finished!')   

    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions

def load_processed_SEEDV_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3, 
                                  n_subs=16, n_vids = 15, n_class=5):
    # input data shape(onesub_onesession):(channels,tot_time) tot_time = sum(eachvids_n_points) 
    # *input data shape（onesub_3session):(channels,tot_time)
    # output : (subs*sum(n_samples_onesub))*channals*time
    #           (16*(sum(n_samples_onesub)))*62*point_len(1250)
    

    list_files = os.listdir(dir)
    list_files = sorted(list_files, key=lambda x: int(re.search(r'\d+', x).group()))
    assert len(list_files) == n_subs
    points_len = int(timeLen*fs)
    points_step = int(timeStep*fs)
    
    # 3 session in all change delete the loop
    file_path = os.path.join(dir,list_files[0])
    onesub_data = sio.loadmat(file_path)  
    n_time = np.squeeze(onesub_data['merged_n_samples_one']).astype(int)
    n_points = np.array(n_time) * fs
    n_samples_onesub = ((n_points-points_len)//points_step+1).astype(int)
    n_samples_sum_onesub = np.sum(n_samples_onesub)
    
    data = np.empty((n_subs*n_samples_sum_onesub,n_chans,points_len),float)

    cnt = 0
    for idx,fn in enumerate(list_files):
        file_path = os.path.join(dir,fn)
        # print(fn)
        onesub_data = sio.loadmat(file_path)     #keys: data,n_points
        EEG_data = onesub_data['merged_data_all_cleaned']   #(channels,tot_n_points_3session)  (60,tot_n_points_3session)
        thr = 30 * np.median(np.abs(EEG_data))
        EEG_data = (EEG_data - np.mean(EEG_data[np.abs(EEG_data)<thr])) / np.std(EEG_data[np.abs(EEG_data)<thr])
        n_points_cum = np.concatenate((np.array([0]),np.cumsum(n_points)))

        
        n_vids_all = n_vids*n_session
        for vid in range(n_vids_all):
            # print('vid:',vid)
            for i in range(n_samples_onesub[vid]):
                # print('sample:',i)
                data[cnt] = EEG_data[:,n_points_cum[vid]+i*points_step:n_points_cum[vid]+i*points_step+points_len]
                cnt+=1
    
    n_samples_onesub = np.array(n_samples_onesub)
    n_samples_sessions = n_samples_onesub.reshape(n_session,-1)
    label = [4, 1, 3, 2, 0] * 3 + [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0] * 2
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions

def load_processed_SEED_NEW_data(dir, fs, n_chans, timeLen, timeStep, n_session=3,
                                 n_subs=15, n_vids=15, n_class=3):
    """
    带缓存机制的 SEED 数据加载函数。
    如果检测到已处理好的标记文件，直接一把加载打包好的大文件；否则重新处理。
    """
    # ================= 1. 定义缓存文件路径 =================
    target_folder_name = f"sliced_len{timeLen}_step{timeStep}_SEED"
    slice_dir = os.path.join(dir, target_folder_name)

    # 你的标记文件路径
    marker_file = os.path.join(slice_dir, 'saved.npy')

    # ================= 2. 检查缓存是否存在且格式兼容 =================
    data_all_path = os.path.join(slice_dir, 'data_all.npy')
    if os.path.exists(marker_file) and os.path.exists(data_all_path):
        print(f" [Fast Load] Found marker file: {marker_file}")
        print(f" Reading slices from: {slice_dir}")

        try:
            onesub_labels = np.load(os.path.join(slice_dir, 'onesub_labels.npy'))
            n_samples_onesub = np.load(os.path.join(slice_dir, 'n_samples_onesub.npy'))
            n_samples_sessions = np.load(os.path.join(slice_dir, 'n_samples_sessions.npy'))

            print(f" Loading data_all.npy (this might take a few seconds due to its large size)...")
            data_all = np.load(os.path.join(slice_dir, 'data_all.npy'))

            if data_all.ndim == 4 and data_all.shape[1] == 1:
                data_all = data_all.squeeze(1)

            print(f" Loaded successfully! Data shape: {data_all.shape}")
            return data_all, onesub_labels, n_samples_onesub, n_samples_sessions

        except Exception as e:
            print(f" Error during loading: {e}")
            raise e

    # ================= 3. 如果缓存不存在，执行原本的处理逻辑 =================
    print(f" Cache not found. Starting processing from scratch...")
    print(f" [SEED Load] Scanning folders 1, 2, 3 in: {dir}")
    print(f" Resampling: Raw(200Hz) -> {fs} Hz | Window: {timeLen}s | Step: {timeStep}s")

    import scipy.signal
    import scipy.io as sio

    # SEED 原始采样率
    ORIG_FS = 200
    points_len = int(timeLen * fs)
    points_step = int(timeStep * fs)

    # 标签映射: 1(Pos)->2, 0(Neu)->1, -1(Neg)->0
    # 实验顺序: 15 个视频的原始标签
    raw_labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
    label_map = {1: 2, 0: 1, -1: 0}
    target_labels = [label_map[l] for l in raw_labels]

    # 初始化容器
    all_slices = []  # 存放所有切片数据 (N, 62, Time)
    all_labels = []  # 存放每个切片的标签
    n_samples_per_vid_list = []  # 记录每个视频切了多少片 (用于 Dataset 索引)

    # 遍历 15 个受试者 (从文件名排序获取)
    session_files = {}
    import re
    for sess in range(1, n_session + 1):
        sess_path = os.path.join(dir, str(sess))
        if not os.path.exists(sess_path):
            raise FileNotFoundError(f"Session folder not found: {sess_path}")

        files = [f for f in os.listdir(sess_path) if f.endswith('.mat')]
        files = sorted(files, key=lambda x: int(re.search(r'^(\d+)_', x).group(1)))

        if len(files) != n_subs:
            raise ValueError(f"Session {sess} has {len(files)} files, expected {n_subs}!")
        session_files[sess] = files

    # --- 双重循环：受试者 -> Session ---
    for sub_idx in range(n_subs):
        print(f"Processing Subject {sub_idx + 1}/{n_subs}...", end='\r')

        for sess in range(1, n_session + 1):
            fn = session_files[sess][sub_idx]
            file_path = os.path.join(dir, str(sess), fn)

            try:
                mat_content = sio.loadmat(file_path)
            except Exception as e:
                print(f"\n Error loading {file_path}: {e}")
                continue

            eeg_keys = [k for k in mat_content.keys() if 'eeg' in k and not 'rf' in k]
            eeg_keys = sorted(eeg_keys, key=lambda x: int(re.search(r'eeg(\d+)', x).group(1)))

            if len(eeg_keys) != n_vids:
                print(f"\n Warning: {fn} has {len(eeg_keys)} keys, expected {n_vids}")
                continue

            for vid_idx, key in enumerate(eeg_keys):
                raw_data = mat_content[key]

                # 1. 降采样
                if fs != ORIG_FS:
                    target_points = int(raw_data.shape[1] * (fs / ORIG_FS))
                    data_resampled = scipy.signal.resample(raw_data, target_points, axis=1)
                else:
                    data_resampled = raw_data

                # 2. Z-score 归一化 (Per Trial)
                mean_v = np.mean(data_resampled, axis=1, keepdims=True)
                std_v = np.std(data_resampled, axis=1, keepdims=True)
                data_norm = (data_resampled - mean_v) / (std_v + 1e-8)

                # 3. 切片
                n_cols = data_norm.shape[1]
                if n_cols < points_len:
                    n_samples_per_vid_list.append(0)
                    continue

                n_wins = (n_cols - points_len) // points_step + 1
                n_samples_per_vid_list.append(n_wins)

                for i in range(n_wins):
                    start = i * points_step
                    end = start + points_len
                    win_data = data_norm[:, start:end]
                    all_slices.append(win_data)
                    all_labels.append(target_labels[vid_idx])

    print(f"\n Data processing finished.")

    # 1. 还原统计数据
    counts_arr = np.array(n_samples_per_vid_list).reshape(n_subs, n_session, n_vids)

    # 2. 每个 session/视频分别取跨受试者的最小窗口数；视频之间仍保留不同长度。
    min_counts = np.min(counts_arr, axis=0)  # Shape: (3, 15)
    print(" [Alignment] Truncating data to match minimum samples across subjects...")

    # 3. 根据最小值过滤数据 (Rebuild the list)
    new_slices = []
    new_labels = []
    current_idx = 0

    for sub in range(n_subs):
        for sess in range(n_session):
            for vid in range(n_vids):
                actual_count = counts_arr[sub, sess, vid]
                keep_count = min_counts[sess, vid]

                segment = all_slices[current_idx: current_idx + keep_count]
                new_slices.extend(segment)

                segment_labels = all_labels[current_idx: current_idx + keep_count]
                new_labels.extend(segment_labels)

                current_idx += actual_count

    # 4. 生成最终数据
    data = np.stack(new_slices, axis=0)
    onesub_labels = np.array(new_labels)
    n_samples_sessions = min_counts
    n_samples_onesub = np.tile(n_samples_sessions.flatten(), n_subs)

    print(f" Final Data Shape: {data.shape}")
    print(f" Metadata Shape Check: n_samples_sessions {n_samples_sessions.shape}")
    print(f" Total Slices: {len(onesub_labels)}")

    # 保存缓存，下次直接读取，避免重复切片
    os.makedirs(slice_dir, exist_ok=True)
    np.save(os.path.join(slice_dir, 'onesub_labels.npy'), onesub_labels)
    np.save(os.path.join(slice_dir, 'n_samples_onesub.npy'), n_samples_onesub)
    np.save(os.path.join(slice_dir, 'n_samples_sessions.npy'), n_samples_sessions)
    np.save(os.path.join(slice_dir, 'data_all.npy'), data)
    np.save(marker_file, [True])
    print(f' Sliced data cached to: {slice_dir}')

    return data, onesub_labels, n_samples_onesub, n_samples_sessions



def save_sliced_data(sliced_data_dir, data, onesub_labels, n_samples_onesub, n_samples_sessions):
    if not os.path.exists(sliced_data_dir+'/metadata'):
        os.makedirs(sliced_data_dir+'/metadata')
    if not os.path.exists(sliced_data_dir+'/data'):
        os.makedirs(sliced_data_dir+'/data')
    np.save(sliced_data_dir+'/metadata/onesub_labels.npy', onesub_labels)
    np.save(sliced_data_dir+'/metadata/n_samples_onesub.npy', n_samples_onesub)
    np.save(sliced_data_dir+'/metadata/n_samples_sessions.npy', n_samples_sessions)
    for sample in range(data.shape[0]):
        np.save(sliced_data_dir+f'/data/data_sample_{sample}.npy', data[sample])
    np.save(sliced_data_dir+'/saved.npy', [True])
    print('save sliced data finished!')
