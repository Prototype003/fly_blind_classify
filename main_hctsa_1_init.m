%% Description

%{

Extract time series features from fly data using hctsa

%}

%% Settings

out_dir = 'hctsa_space_subtractMean_removeLineNoise/';

%% Load

source_dir = 'data/preprocessed/';
source_file = 'fly_data_subtractMean_removeLineNoise';

loaded = load([source_dir source_file]);
data = loaded.data;

%% Setup for HCTSA - training set
% Reformat into series x time matrix

% Training dataset
train_set = data.train; % time x channels x trials x flies x conditions
% Get labels for each time-series
%   dimensions - (channels x trials x flies x conditions)
dims = size(train_set);
train_ids = cell(dims(2:end)); % details of each time-series
for ch = 1 : dims(2)
    for tr = 1 : dims(3)
        for f = 1 : dims(4)
            for c = 1 : dims(5)
                train_ids{ch, tr, f, c} = ['fly' num2str(f) ',channel' num2str(ch) ',epoch' num2str(tr) ',condition' num2str(c)];
            end
        end
    end
end
% Reformat to (series x time)
train_set = permute(train_set, [2 3 4 5 1]); % channels x trials x flies x conditions x time
train_set = reshape(train_set, [prod(dims(2:end)) dims(1)]); % Collapse all dimensions other than time
train_ids = reshape(train_ids, [prod(dims(2:end)) 1]); % Collapse labels also
% Create hctsa matrix
timeSeriesData = train_set;
labels = train_ids; % keywords are already unique
keywords = train_ids;
save([out_dir 'train.mat'], 'timeSeriesData', 'labels', 'keywords');
tic;
TS_Init([out_dir 'train.mat'], [], [], [true, false, false], [out_dir 'HCTSA_train.mat']);
toc
disp('training set done');

%% Setup for HCTSA - validation1 set
% Reformat into series x time matrix

% Training dataset
validate1_set = data.validate1; % time x channels x trials x flies x conditions
% Get labels for each time-series
%   dimensions - (channels x trials x flies x conditions)
dims = size(validate1_set);
validate1_ids = cell(dims(2:end)); % details of each time-series
for ch = 1 : dims(2)
    for tr = 1 : dims(3)
        for s = 1 : dims(4)
            validate1_ids{ch, tr, s} = ['segment' num2str(s) ',channel' num2str(ch) ',epoch' num2str(tr)];
        end
    end
end
% Reformat to (series x time)
validate1_set = permute(validate1_set, [2 3 4 1]); % channels x trials x flies x conditions x time
validate1_set = reshape(validate1_set, [prod(dims(2:end)) dims(1)]); % Collapse all dimensions other than time
validate1_ids = reshape(validate1_ids, [prod(dims(2:end)) 1]); % Collapse labels also
% Create hctsa matrix
timeSeriesData = validate1_set;
labels = validate1_ids; % keywords are already unique
keywords = validate1_ids;
save([out_dir 'validate1.mat'], 'timeSeriesData', 'labels', 'keywords');
tic;
TS_Init([out_dir 'validate1.mat'], [], [], [true, false, false], [out_dir 'HCTSA_validate1.mat']); % ~470s
toc
disp('validation1 set done');
