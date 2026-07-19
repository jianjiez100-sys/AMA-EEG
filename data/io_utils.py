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
    print(f"🚀 [One-Step Load] Reading FACED .pkl from: {dir}")
    print(f"📉 Resampling: 250 Hz -> {fs} Hz | Window: {timeLen}s")

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
            print(f"❌ Error loading {fn}: {e}")
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

    print(f"✅ Data loaded successfully. Output Shape: {data.shape}")
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
                                  n_subs=15, n_vids = 15, n_class=3):
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
    label =  list(np.array([1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1])+1) * 3
    onesub_labels = []
    for i in range(len(label)):
        onesub_labels = onesub_labels + [label[i]]*n_samples_onesub[i]   
    return data, np.array(onesub_labels), n_samples_onesub, n_samples_sessions



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

def test_load_processed_SEEDV_data():
    data_dir = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sxk'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 250
    n_channs = 62
    n_session = 3

    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_labels)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_labels'] = onesub_labels
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

def test_load_processed_SEEDV_NEW_data():
    data_dir = '/mnt/data/model_weights/grm/SEEDV-NEW/processed_data'
    # data_dir = '/mnt/data/model_weights/grm/SEEDV_new2/processed_ddata'
    # data_dir = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sxk'
    data_dir2 = '/mnt/data/model_weights/grm/SEEDV/EEG_processed_sampled'
    # data_dir2 = 'D:/graduate/G2/xinke/SEEDV/EEG_processed_sampled'
    timeLen = 5
    timeStep = 2
    fs = 125
    n_channs = 60
    n_session = 3

    data, onesub_labels, n_samples_onesub, n_samples_sessions = load_processed_SEEDV_NEW_data(data_dir,fs,n_channs,timeLen,timeStep,n_session)
    print(data.shape)
    print(onesub_labels)
    print(n_samples_onesub)
    print(n_samples_sessions)
    sampled_data = {}
    sampled_data['data'] = data
    sampled_data['onesub_labels'] = onesub_labels
    sampled_data['n_samples_onesub'] = n_samples_onesub
    sampled_data['n_samples_sessions'] = n_samples_sessions

if __name__ == '__main__':
    # test_load_processed_SEEDV_data()
    test_load_processed_SEEDV_NEW_data()