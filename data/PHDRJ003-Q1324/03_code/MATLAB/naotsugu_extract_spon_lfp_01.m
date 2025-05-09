%% naotsugu_extract_spon_lfp_01.m
% File made 18/09/2020. 

% It will work on the RDM collection
% '\\uq.edu.au\uq-inst-gateway1\phdrj003-q1324'

% The purpose of this file is to: 
% 1) Use pre-2017 frequency-sweep dataset with N = 12,
% 2) Load 'LFP.mat' structure which contains the visual flicker indexes.
% 3) Find 18 seconds before the first visual flicker, to have spontaneous
% lfp data after 3 minutes of isoflurane exposure.

% This file was edited from 'process_lfp_trials_01_ICA_04'.

%% NOTES
% 18/09/2020 RJ File made. Edits from the original file include renaming
% variables for the sake of legibility. Made more dynamic references to
% paths. Removed unnecessary cd() to X:/ directories. Files are loaded from 
% the ICA directories as they are more recent.
% 10/06/2022 RJ Commented out the condition which only loads flies with 3
% conditions into the table. Loads all flies into the table instead.

% - Rhiannon Jeans 2020
%% Script start

clear all; close all;% clc;

%% Load your flies all together

% These switches are to load the correct 'LFP.mat' structure later on
ExtraRecovery = 1; % remove extra recovery condition for those flies that have it (trial number 150+)
FilterStatus=1;
SVDStatus = 0; % Turn off single value decomposition

% This is the RDM collection path
%{
folder_root = 'X:\02_processed_data\';
code_path = 'X:\03_code\MATLAB';
addpath(code_path)
addpath(genpath('X:\03_code\MATLAB\functions\'))

fly_list = dir([folder_root 'Analyzed_*ICA']);
%}
folder_root = '..\..\02_processed_data\'; % relative to here
code_path = '.\'; % we're here
% adding code and function paths might not be needed for this script?
fly_list = dir([folder_root 'Analyzed_*']); % will only have the prepared data

if length(fly_list) ~= 18
    error('fly_list length is not 18.')
elseif length(fly_list) == 18
    disp(['Total length of fly_list is ' num2str(18)])
end % length check

remove_fields = {'date', 'bytes', 'isdir', 'datenum'};
fly_list = rmfield(fly_list, remove_fields);

for fly = 1:length(fly_list)

    varSaveList = who;
    varSaveList = [varSaveList; {'varSaveList'}]; %A bit silly, but this is needed to prevent the list itself from being wiped
    varSaveList = [varSaveList; {'fly'; 'fly_list'}]; %This is to prevent the iterator from being wiped
    
    folder_name = [folder_root fly_list(fly).name];
    disp(folder_name)
    
    disp('Using regexprep to find root filename...')
    
    % replace '_ICA'
    filename = [fly_list(fly).name];
    pattern = '_ICA';
    replacement = '';
    pre_ICA_filename = regexprep(filename,pattern,replacement); % chop off the _Analyzed part of the filename

    pre_ICA_folder_name = [folder_root pre_ICA_filename];
    
    fprintf('Adding data to the workspace.\n');
    data_dir = [pre_ICA_folder_name filesep 'SpliceData' filesep 'AllSplicedDataFlyBlock_1.mat'];
    stim_dir = [pre_ICA_folder_name filesep 'SpliceData' filesep 'AllSplicedStimulusDataBlock_1.mat'];
    
    % Check files exist in their specified locations
    if isfile(data_dir)
        disp([data_dir ' is found on the path.']);
    elseif ~isfile(data_dir)
        error([data_dir ' is not found on the path.']);
    end % if isfile
    %{
        % Check files exist in their specified locations
    if isfile(stim_dir)
        disp([stim_dir ' is found on the path.']);
    elseif ~isfile(stim_dir)
        error([stim_dir ' is not found on the path.']);
    end % if isfile
    %}
    load_check = 0;
    while load_check ==0
        try
            load(data_dir);
            %{
            load(stim_dir);            
            %}
            load_check=1;
        catch
            load_check =0;
            warning('Error loading data. Trying again in 30 seconds.')
            pause(30);
        end
    end
    
    % Load the LFP.mat structure
    lfp_dir = [ folder_name filesep  'LFP_SVDStat' num2str(SVDStatus) 'FiltStatus_' num2str(FilterStatus)];
    
    load_check = 0;
    while load_check ==0
        try
            load([lfp_dir filesep 'LFP.mat'], 'LFP');
            load_check=1;
        catch
            load_check =0;
            warning('Error saving data. Trying again in 30 seconds.')
            pause(30);
        end % try
    end % while
    
    fs = 1000; % sampling rate

    % Make a note of the isoflurane conditions in this experiment
    isoflurane_fields = unique(extractfield(LFP,'TrialType'),'stable'); % stable arguments means 'sort' is not applied to the order
    
    spon_dat = struct;
    % To find the first visual stimulus of each condition, loop over
    % isoflurane_fields and extract those parts out of the structure
    for iso_id = 1:length(isoflurane_fields)
       temp = LFP(arrayfun(@(x) strcmp(x.TrialType, isoflurane_fields(iso_id)), LFP)); 
       
       % Extract 18 seconds before the first visual stimulus. First check if
       % there are obvious airpuff artifacts in a minute before the stimulus
       % start
       spon_dat(iso_id).first_visual = temp(1).Index(1,1);
       spon_dat(iso_id).eighteen_secs_before_visual = spon_dat(iso_id).first_visual - 18*fs;
       spon_dat(iso_id).selection_index = spon_dat(iso_id).eighteen_secs_before_visual : spon_dat(iso_id).first_visual;
       
       spon_dat(iso_id).pre_visual_lfp = SplicedData(:, spon_dat(iso_id).selection_index);
       spon_dat(iso_id).trial_type = isoflurane_fields(iso_id);
    
    end % iso_id
    
    % Some notes if you want to visually inspect the data
    % check between pre-ICA/ filter index and LFP post-processed index
    %     plot(smooth(SplicedData(13,first_visual:first_visual+2000)))
    %     hold on
    %     plot(LFP(1).P1On(13,1:2000))
    
    % check locus of the airpuff based on a movement artifact (rough check)
%     min_before_visual = first_visual-60*1000:first_visual;
%     plot(SplicedData(13,min_before_visual))

    %% Save the pre_visual_lfp data
    save_dir = [pre_ICA_folder_name filesep  'Naotsugu' filesep];
    
    if ~isdir(save_dir)
        mkdir(save_dir)
    end % check isdir
    
    output_filename = 'pre_visual_lfp.mat';
    
    save_check = 0;
    while save_check == 0
        try
            %{
            save([save_dir filesep output_filename], 'pre_visual_lfp', 'fs','spon_dat','pre_ICA_filename');
            %}
            % pre_visual_lfp doesn't exist outside of spon_dat
            save([save_dir filesep output_filename], 'fs','spon_dat','pre_ICA_filename');
            disp(['Saved ' output_filename '!'])
            save_check = 1;
        catch
            save_check = 0;
            warning(['Error saving ' output_filename '. Trying again in 30 seconds.'])
            pause(30);
        end % try
    end % while
    
    disp('Clearing all variables.')
    clearvars('-except',varSaveList{:}) %Clear everything except initialisation variables

end % fly loop

%% Next, load specific fly data and load into an output table with the Wake and Iso data.

merge_table = struct;
row_count = 0;
fly_count = 0;

for fly = 1:length(fly_list)

    varSaveList = who;
    varSaveList = [varSaveList; {'varSaveList'}]; %A bit silly, but this is needed to prevent the list itself from being wiped
    varSaveList = [varSaveList; {'fly'; 'fly_list'}]; %This is to prevent the iterator from being wiped
    
    folder_name = [folder_root fly_list(fly).name];
    disp(folder_name)
    
    disp('Using regexprep to find root filename...')
    
    % replace '_ICA'
    filename = [fly_list(fly).name];
    pattern = '_ICA';
    replacement = '';
    pre_ICA_filename = regexprep(filename,pattern,replacement); % chop off the _Analyzed part of the filename

    pre_ICA_folder_name = [folder_root pre_ICA_filename];
    
    load_dir = [pre_ICA_folder_name filesep  'Naotsugu' filesep];
    input_filename = 'pre_visual_lfp.mat';

    file_load = [load_dir input_filename];
    
    % Check files exist in their specified locations
    if isfile(file_load)
        disp([file_load ' is found on the path.']);
    elseif ~isfile(file_load)
        error([file_load ' is not found on the path.']);
    end % if isfile
    
    load_check = 0;
    while load_check ==0
        try
            load(file_load);
            disp(['Successfully loaded ' file_load ]);
            load_check=1;
        catch
            load_check =0;
            warning('Error loading data. Trying again in 30 seconds.')
            pause(30);
        end % try
    end % while

  
    fly_count = fly_count + 1;
    
    field_list = fieldnames(spon_dat);
    
    for y = 1:length(spon_dat)
        row_count = row_count + 1;
        for x = 1:length(field_list)
            merge_table(row_count).(field_list{x}) = spon_dat(y).(field_list{x});
        end % x length field_list
        
        % Label flies depending on number of conditions
        if length(spon_dat) > 3
            merge_table(row_count).fly_group = 2;
            
        elseif length(spon_dat) == 3
            merge_table(row_count).fly_group = 1;
        end % if
    
        
        merge_table(row_count).fly_number = fly_count;
        merge_table(row_count).fly_label = pre_ICA_filename;
        
    end % y length spon_dat
    
end  % fly

%{
naotsugu_folder_root = [folder_root filesep 'Naotsugu' filesep];
%}
% folder_root already has filesep at the end
naotsugu_folder_root = [folder_root 'Naotsugu' filesep];

if ~isdir(naotsugu_folder_root)
    mkdir(naotsugu_folder_root)
end % isdir

output_filename = 'merge_table.mat';

save_check =0;
while save_check ==0
    try
        save([naotsugu_folder_root output_filename], 'merge_table', '-v7.3');
        disp(['Saved ' output_filename ' to ' naotsugu_folder_root '.'])
        save_check = 1;
    catch
        warning(['Error saving ' output_filename ' to ' naotsugu_folder_root '.'])
        save_check = 0;
        pause(30);
    end % try
end % while
