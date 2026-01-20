decision_window_list = [0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 1.5625];
decision_window_name_list = '01, 025, 05, 075, 1, 2, 5, 10, 015625';
decision_window_name_list = strsplit(decision_window_name_list, ', ');

for decision_window_ID = 1:8
    window_duration = decision_window_list(decision_window_ID);  % 时间窗长度/s
    window_name = decision_window_name_list{decision_window_ID}; % 时间窗名称
    Data_Path = ''; % 数据集路径
    data_save_path = ['',window_name,'s\']; % EEG数据存储数据
    for subject_ID = 1:16
        %% 数据及重要索引读取
        sampling_rate = 128; % 数据集采样率/Hz
        all_data = load([Data_Path,'S',num2str(subject_ID),'.mat']);
        trials = all_data.trials;  % 读取各trials的数据
        num_trials = size(trials,2); % trails的总数
        raw_EEG_data = []; % 存储EEG数据
        attended_ear_label = []; % 存储各sample的关注耳朵标签
        cue_start = 1; % 起始cue，用来统计听觉任务标签的cue
        for trial = 1:num_trials
            raw_trail_EEG_data = all_data.trials{trial}.RawData.EegData;
            attended_ear_trail = all_data.trials{trial}.attended_ear; % trail关注的耳朵
            raw_EEG_data = [raw_EEG_data; raw_trail_EEG_data]; % 拼接得到完整EEG数据矩阵
            cue_end = size(raw_EEG_data,1); % 结束cue
            if attended_ear_trail == 'L'
                attended_ear_label(cue_start:cue_end,1) = 1;  % 左耳 == 1
            else
                attended_ear_label(cue_start:cue_end,1) = 2; % 右耳 == 2
            end
            cue_start = cue_end + 1;
        end
        
        %% EEG数据重参考+滤波 (需要EEGlab工具箱，将其添加至matlab函数库中)
        raw_EEG_data = raw_EEG_data';
        train_EEG = pop_importdata('setname','train', ...
        'data', raw_EEG_data, ...
        'dataformat', 'array', ...
        'srate', sampling_rate, ...
        'nbchan', 64);
        train_EEG = pop_reref(train_EEG, []); % 进行平均重参考
        train_EEG = pop_eegfilt(train_EEG, 1 ,32, [], 0, 0, 0,'fir1', 0); % 进行FIR滤波
        raw_EEG_data = train_EEG.data;
        raw_EEG_data = raw_EEG_data';
        
        %% 根据 decision_window 大小分割数据集，制作样本
        sample_num = size(raw_EEG_data,1); % 读取采样点个数
        window_size = round(sampling_rate * window_duration); % 裁切窗口大小
        
        % 预计算可能的样本数量
        total_samples_possible = floor(size(raw_EEG_data, 1) / window_size);
        cnt_subject = zeros(total_samples_possible, window_size, 64);
        
        % 分割数据
        current_sample = 1; % 用于记录当前样本的索引
        for start_idx = 1:window_size:size(raw_EEG_data, 1)
            end_idx = start_idx + window_size - 1;
            if end_idx > size(raw_EEG_data, 1)
                break;
            end
            cnt_subject(current_sample, :, :) = raw_EEG_data(start_idx:end_idx, :); % 保存数据
            label_subject(current_sample,1) = attended_ear_label(start_idx); % 保存label
            current_sample = current_sample + 1;  % 当前样本索引
        end
        
        %% 保存数据
        cnt_save_name = ['data_', int2str(subject_ID)]; % cnt数据保存名称
        label_save_name = ['label_', int2str(subject_ID)]; % label数据保存名称
        eval([cnt_save_name,'=cnt_subject',';']); % 将字符串转换为matlab可执行语句
        eval([label_save_name,'=label_subject',';']); % 将字符串转换为matlab可执行语句
        save([data_save_path,'\data_', int2str(subject_ID),'.mat'],['data_', int2str(subject_ID)]);
        save([data_save_path,'\label_', int2str(subject_ID),'.mat'],['label_', int2str(subject_ID)]);
    end
end