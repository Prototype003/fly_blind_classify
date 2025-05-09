%% Extract 18 seconds
% Extract 18 seconds before and after sleep start time to divide single
% sleep epochs into wake and sleep.
%% NOTES
% 18/06/2022 RJ Created new file.
% 28/08/2023 AL
%   Replace Z:\ with ..\..\
%   Extract 18s before sleep start time
%   Extract 18s, from 5 minutes after sleep start time

%% Script Start

fly_list = {'27072018_SponSleep_24hrs';
'14092018_SponSleep_LFP'; 
'17092018_SponSleep_LFP';
'19092018_SponSleep_LFP';
'30102018_SponSleep_LFP';
'01112018_SponSleep_LFP';
'03112018_SponSleep_LFP';
'13112018_SponSleep_LFP';
'21112018_SponSleep_LFP';
'22112018_SponSleep_LFP';
'28112018_SponSleep_LFP';
'11122018_SponSleep_LFP';
'13122018_SponSleep_LFP';
'18122018_SponSleep_LFP';
'10012019_SponSleep_LFP';
'22012019_SponSleep_LFP';
'20022019_SponSleep_LFP';
'06032019_SponSleep_LFP';
'13032019_SponSleep_LFP'}

blocks_to_extract = [2,2,1,5,5,3,5,6,2,3,5,4,5,4,3,3,8,8,7];

%% Extract 18 seconds per fly
LFP_data = struct;

count = 0;

for fly = 1:length(fly_list)
    
    %% Load fly data
    datasetname = fly_list{fly}
    epoch_path = ['..\..\02_processed_data\LFP\' datasetname '\trimmed_LFP_200Hz.mat'];
    if strcmp(datasetname, '14092018_SponSleep_LFP')
        resample_path = ['..\..\02_processed_data\LFP\' datasetname '\LFP\Analyzed_Block-1\' datasetname '_chunk_0' num2str(blocks_to_extract(fly)) '_1000Hz.mat'];
    else
        resample_path = ['..\..\02_processed_data\LFP\' datasetname '\LFP\Analyzed_Block-' num2str(blocks_to_extract(fly)) '\' datasetname '_chunk_0' num2str(blocks_to_extract(fly)) '_1000Hz.mat'];
    end
    
    disp('Loading epoch data.');
    load_val = 0;
    while load_val == 0
        try
            epoch = load(epoch_path);
            load_val = 1;
        catch
            load_val = 0;
            warning('Error loading data. Trying again in thirty seconds.')
            pause(10)
        end % try
    end % while
    
    epoch = epoch.trimmed_LFP.epoch;
    
    for x = 1:length(epoch)
        epoch(x).block_num = epoch(x).unique_block_num(1,1);
    end % 
    
    blocks = extractfield(epoch, 'block_num');
    index = 1:length(blocks);
    sleep_block = index(...
        blocks == blocks_to_extract(fly) &...
        [epoch.epoch_dur_mins] >= 5 &... % at least 5 minutes
        strcmp('Inactive', {epoch.status}) &... % make sure it's sleeping
        cellfun(@(x) length(x)==1, {epoch.unique_block_num})); % make sure the block doesn't go into another block
    
    if strcmp(datasetname, '14092018_SponSleep_LFP') | strcmp(datasetname, '13112018_SponSleep_LFP') | strcmp(datasetname, '13032019_SponSleep_LFP')
        % RJ's selection
        block_select = sleep_block(2);
    else
        % Try get longest block which meets criteria
        [max_dur, b] = max([epoch(sleep_block).epoch_dur_mins]);
        block_select = sleep_block(b);
        
        % Get the first block which meets criteria
        %block_select = sleep_block(1);
    end
    
    epoch2 = epoch; % backup variable for debugging
    epoch = epoch(block_select);
    disp(['fly ' num2str(fly) ': selected block length ' num2str(epoch.epoch_dur_mins) 'mins']);
    
    %% remove fields
    fields_to_remove = {'start_unix_idx', 'end_unix_idx', 'vid_frames', 'contours','delta_pixels','delta_pixels_prop', 'mean_contours','mean_prop_pixels', 'delta_pixels','delta_pixels_prop', 'lfp_start','lfp_end','lfp_idx_start','lfp_idx_end'};
    epoch = rmfield(epoch,fields_to_remove);
    
    %% Load 1000Hz data    
    disp('Loading 1000Hz data')
    load_val = 0;
    while load_val == 0
        try
            LFP = load(resample_path);
            load_val = 1;
        catch
            load_val = 0;
            warning('Error loading data. Trying again in thirty seconds.')
            pause(10)
        end % try
    end % while
    
    times = LFP.EEG.epoch_times;
    LFP = LFP.EEG.data;
    
%% Notes

%{

epoch.start_unix is the same as epoch.start_time
    datetime(epoch.start_unix, 'ConvertFrom', 'posixtime')
epoch.end_unix is the same as epoch.end_time

times (LFP.EEG.epoch_times) specifies a time range longer than
    epoch.start_unix to epoch.end_unix

start_sleep is set as epoch.start_unix
    So, epoch.start_unix and epoch.end_unix specify onset of sleep
        and resumption of physical activity? And epoch.dur, epoch_dur_mins
        refer to the duration of the sleep bout?
    Check whether:
        A) Is this onset of sleep set as AFTER 5 minutes of inactivity?
        B) Is this onset the start of any bout of inactivity longer than 5
            minutes?
            If B), then flies are moving during "wake" recording
                (regardless of whether it's the 18s immediately preceding
                    sleep "onset" or 18s 5 minutes before "onset")
    BUT - why do all epoch blocks have start_unix and end_unix, regardless
        of label 'inactive' or 'active'?

%}

    %% Find pre and post sleep times
    eighteen = 18*1000; % 18 seconds at 1000Hz
    start_sleep = epoch.start_unix;
    end_sleep = epoch.end_unix;
       
    [val,idx] = min(abs(times - start_sleep));
    unix_start = times(1,idx); % sanity check
    
    [val, idx_end] = min(abs(times - end_sleep));
    
    %% Times
    
    %{
    wake = LFP(:, idx-eighteen:idx);
    sleep = LFP(:, idx:idx+eighteen);
    %}
    
    minute = 60 * 1000; % 1 minutes 1 1000Hz
    fiveMinutes = 60 * 5 * 1000; % 5 minutes at 1000Hz
    tenMinutes = 60 * 10 * 1000; % 10 minutes at 1000Hz
    
    periods = {'wakeEarly', 'wake', 'sleepEarly', 'sleepLate'};
    period_idx = {...
        idx-(1*minute)-eighteen:idx-(1*minute),...
        idx-eighteen:idx,...
        idx+(2*minute)-eighteen:idx+(2*minute),...
        idx_end-(1*minute)-eighteen:idx_end-(1*minute)};
    
    % If sleepLate period starts before 5 minutes,
    %   get period immediately from 5 minutes
    if idx_end-(1*minute)-eighteen < idx+5*minute
        period_idx{end} = idx+5*minute : idx+5*minute+eighteen;
    end
    
    %% Check
    
    % Check that we have at least 10 minutes of sleep
    if epoch.epoch_dur_mins < 10
        warning(['fly' num2str(fly) ' ' datasetname ' sleep duration <10 minutes (' num2str(epoch.epoch_dur_mins) 'mins)']);
    end
    
    % Sum squared voltages over 18s as proxy measure of movement
    LFP_summed = movsum(LFP.^2, eighteen, 2);
    
    fig_dir = 'extract_sleep_naotsugu_01\';
    if ~exist(fig_dir, 'dir')
        mkdir(fig_dir);
    end
    
    fig = figure('visible', 'off', 'Units', 'normalized', 'Position', [0 0 1 1]);
    imagesc(LFP_summed); c = colorbar;
    ylabel('ch');
    xlabel('t');
    title(c, '\SigmaV^2');
    title(['fly' num2str(fly) ' ' datasetname], 'interpreter', 'none')
    set(gca, 'XTick', [idx-fiveMinutes idx idx+fiveMinutes idx+tenMinutes]);
    set(gca, 'XTickLabel', {'-5mins', '0mins', '5mins', '10mins'});
    saveas(fig, [fig_dir 'fly' num2str(fly) '_' datasetname], 'png');
    hold off
    close(fig);
    
    fig = figure('visible', 'off', 'Units', 'normalized', 'Position', [0 0 1 1]);
    imagesc(LFP_summed(:, period_idx{1}(1) : idx_end)); c = colorbar;
    ylabel('ch');
    xlabel('t');
    title(c, '\SigmaV^2');
    title(['fly' num2str(fly) ' ' datasetname ' zoomed'], 'interpreter', 'none')
    set(gca, 'XTick', [idx-period_idx{1}(1) idx-period_idx{1}(1)+5*minute idx_end-period_idx{1}(1)]);
    set(gca, 'XTickLabel', {'0mins', '5mins', [num2str(epoch.epoch_dur_mins) 'mins']});
    hold on;
    channels = [0.5 16.5];
    lwidth = 1;
    for p = 1 : length(periods)
        line([period_idx{p}(1)-period_idx{1}(1) period_idx{p}(1)-period_idx{1}(1)], channels, 'LineWidth', lwidth, 'Color', 'r');
        line([period_idx{p}(end)-period_idx{1}(1) period_idx{p}(end)-period_idx{1}(1)], channels, 'LineWidth', lwidth, 'Color', 'r');
    end
    saveas(fig, [fig_dir 'fly' num2str(fly) '_' datasetname '_zoomed'], 'png');
    close(fig);
    
    %% Extract times
    
    for p = 1 : length(periods)
        count = count + 1;
        LFP_data(count).filename = datasetname;
        LFP_data(count).fly_num = fly;
        LFP_data(count).LFP = LFP(:, period_idx{p});
        LFP_data(count).index = period_idx{p};
        LFP_data(count).times = times(period_idx{p});
        LFP_data(count).epoch_idx = block_select;
        LFP_data(count).block_num = blocks_to_extract(fly);
        LFP_data(count).status = periods{p};
        LFP_data(count).epoch_ref = epoch;
    end
    %{
% wake_early
    LFP_data(count).filename = datasetname;
    LFP_data(count).fly_num = fly;
    LFP_data(count).LFP = wake;
    LFP_data(count).index = idx-fiveMinutes:idx-fiveMinutes+eighteen;
    LFP_data(count).times = times(idx-fiveMinutes:idx-fiveMinutes+eighteen);
    LFP_data(count).epoch_idx = block_select;
    LFP_data(count).block_num = blocks_to_extract(fly);
    LFP_data(count).status = 'Active5';
    LFP_data(count).epoch_ref = epoch;
    
    count = count + 1;
% wake
    LFP_data(count).filename = datasetname;
    LFP_data(count).fly_num = fly;
    LFP_data(count).LFP = wake;
    LFP_data(count).index = idx-eighteen:idx;
    LFP_data(count).times = times(idx-eighteen:idx);
    LFP_data(count).epoch_idx = block_select;
    LFP_data(count).block_num = blocks_to_extract(fly);
    LFP_data(count).status = 'Active';
    LFP_data(count).epoch_ref = epoch;
    
    count = count + 1;
% sleep
    LFP_data(count).filename = datasetname;
    LFP_data(count).fly_num = fly;
    LFP_data(count).LFP = sleep;
    LFP_data(count).index = idx+fiveMinutes-eighteen:idx+fiveMinutes;
    LFP_data(count).times = times(idx+fiveMinutes-eighteen:idx+fiveMinutes);
    LFP_data(count).epoch_idx = block_select;
    LFP_data(count).block_num = blocks_to_extract(fly);
    LFP_data(count).status = 'Inactive5';
    LFP_data(count).epoch_ref = epoch;
    
    count = count + 1;
% 'deep' sleep
    LFP_data(count).filename = datasetname;
    LFP_data(count).fly_num = fly;
    LFP_data(count).LFP = sleep2;
    LFP_data(count).index = idx+tenMinutes-eighteen:idx+tenMinutes;
    LFP_data(count).times = times(idx+tenMinutes-eighteen:idx+tenMinutes);
    LFP_data(count).epoch_idx = block_select;
    LFP_data(count).block_num = blocks_to_extract(fly);
    LFP_data(count).status = 'Inactive10';
    LFP_data(count).epoch_ref = epoch;
%}
    clearvars -except fly_list blocks_to_extract LFP_data count fly

end

save_check =0;
while save_check ==0
    try
        save('..\..\02_processed_data\Naotsugu\LFP_data.mat', 'LFP_data', '-v7.3');
        disp(['Saved file.'])
        save_check = 1;
    catch
        warning(['Error saving data. Trying again in 30 seconds.'])
        save_check = 0;
        pause(30);
    end % try
end % while


