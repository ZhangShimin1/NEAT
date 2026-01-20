decision_window_list = [0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10];
decision_window_name_list = '01, 025, 05, 075, 1, 2, 5, 10';
decision_window_name_list = strsplit(decision_window_name_list, ', ');

for decision_window_ID = 1:8
    window_duration = decision_window_list(decision_window_ID);  % 时间窗长度/s
    window_name = decision_window_name_list{decision_window_ID}; % 时间窗名称
    data_save_path = ['D:\EEG Audio Dataset\Processed Dataset\DTU Data\128Hz\', window_name, 's']; % EEG数据存储数据
    for subject_ID = 1:18
        %% 数据及重要索引读取
        Data_Path = 'D:\EEG Audio Dataset\Raw Dataset\DTU Data\EEG\'; % 数据集路径
        sampling_rate = 512; % 数据采样率/Hz
        down_sampling_rate = 128;  % 降采样频率/Hz
        trial_time = 50; % 每个trial的持续时间/s
        all_data = load([Data_Path,'S',num2str(subject_ID),'.mat']);
        raw_EEG_data = all_data.data.eeg{1}; % 读取原始EEG数据
        raw_EEG_data = raw_EEG_data(:,1:66);
        attend_mf_label = all_data.expinfo.attend_mf; % 听觉注意Speaker标签
        attend_lr_label = all_data.expinfo.attend_lr; % 听觉注意左右耳标签
        trial_trigger = all_data.expinfo.trigger; % 事件触发编码
        n_speakers = all_data.expinfo.n_speakers; % 参与说话个数，其==2为有效trial
        sample_cue = all_data.data.event.eeg.sample; % 每个trial起始采样点数
        sample_trigger = all_data.data.event.eeg.value; % 每个trial的对应触发编码
        sample_trigger = cell2mat(sample_trigger);
        
        %% EEG数据重参考+滤波 (需要EEGlab工具箱，将其添加至matlab函数库中)
        raw_EEG_data = raw_EEG_data';
        train_EEG = pop_importdata('setname','train', ...
        'data', raw_EEG_data, ...
        'dataformat', 'array', ...
        'srate', 512, ...
        'nbchan', 66);
        train_EEG = pop_reref(train_EEG, []); % 进行平均重参考
%         train_EEG = pop_eegfilt(train_EEG, 1, 32); % 进行FIR滤波
        train_EEG = pop_resample(train_EEG, down_sampling_rate);
        raw_EEG_data = train_EEG.data;
        raw_EEG_data = raw_EEG_data';
    
        %% 删除非听觉任务的Trial：n_speakers == 1
        id = n_speakers(:) == 1; % 提取非任务的trial编号
        attend_mf_label(id) = [];
        attend_lr_label(id) = [];
        trial_trigger(id) = [];
        
        %% 根据trial_trigger读取听觉注意任务的索引
        trial_ID = zeros(1, length(trial_trigger)); % 初始化结果向量
        startIndex = 1; % 初始化搜索的起始索引
        for i = 1:length(trial_trigger)
            foundIndex = find(sample_trigger(startIndex:end) == trial_trigger(i), 1) + startIndex - 1;
            if isempty(foundIndex) % 检查是否找到索引
                error('未找到匹配的索引。');
            else
                trial_ID(i) = foundIndex;
                startIndex = foundIndex + 1;
            end
        end
        trial_cue = sample_cue(trial_ID); % 听觉任务的cue
        adjusted_trial_cue = round(trial_cue / 4);
    
        %% 分割数据集
        trial_num = size(adjusted_trial_cue,1); % 读取trial个数
        window_size = round(down_sampling_rate * window_duration); % 裁切窗口大小
        window_sample_num = round(trial_time/window_duration); % 窗内样本的总个数
        
        n_sample = 1; % 初始化样本索引值
        cnt_subject = [];
        label_subject = [];
        CUE = [];
        for trial = 1:trial_num
            cue = adjusted_trial_cue(trial); % 索引值
            label_speaker = attend_mf_label(trial); % 关注的说话者label
            label_ear = attend_lr_label(trial); % 关注的耳朵label
            for sample = 1:window_sample_num
                I = raw_EEG_data(cue:cue+window_size-1,:);
                cnt_subject(n_sample, :, :) = I;
                label_subject(n_sample, :, :) = label_ear;
                CUE(n_sample) = cue; % 保存cue，看代码是否正确
                cue = cue + window_size;
                n_sample = n_sample + 1;
            end
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