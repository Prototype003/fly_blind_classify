%% Description

%{

Apply thresholds obtained from test dataset to validation dataset(s)

%}

%% Settings

val_set = 'multidose';
val_suffix = 'BatchNormalised';%'BatchNormalised'; % empty or 'BatchNormalised'
val_string = [val_set val_suffix '_mdSplit']; % 'xxxxBatchNormalised'; '_mdSplit' only for batch-normalised

%class_type = 'nearestMean'; % nearest mean classification
class_type = 'nearestMedian'; % nearest median classification

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['../hctsa_space' preprocess_string '/'];
source_prefix = ['HCTSA_' val_set];

pred_dir = ['results' preprocess_string '/'];
pred_file = ['class_' class_type '_' val_string];

out_dir = ['results' preprocess_string '/'];
%out_file = ['class_' class_type '_' val_string '_accuracy.mat'];

addpath('../');
here = pwd;
cd('../'); add_toolbox; cd(here);

% Separate out first 8 and last 4 flies
flyGroups = {(1:8), (9:12)};

%% Load predictions

tic;
load([pred_dir pred_file]);
toc

%% Load labelled data to get labels
tic;
label_dir = ['../hctsa_space' preprocess_string '/'];
label_file = [source_prefix '_channel1.mat']; % assumes all channels have same order of epochs

labelled_matfile = matfile([label_dir label_file]);
data_labelled = labelled_matfile.TimeSeries;
data_labels = data_labelled.Keywords;
toc

%% Find associated rows for each group of flies

flyGroup_rows = cell(size(flyGroups));

% Emulate hctsa TimeSeries struct (with field: keywords)
TimeSeries = struct();
TimeSeries.Keywords = data_labels;

for fgroup = 1 : length(flyGroups)
    
    flyGroup_rows{fgroup} = zeros(size(data_labels));
    
    for fly = flyGroups{fgroup}
        flyGroup_rows{fgroup} = flyGroup_rows{fgroup} | getIds({['fly' num2str(fly)]}, TimeSeries);
    end
    
end

%% Get and interpret condition keyword
% Two conditions - wake and unawake
% Assume (currently, from trained models):
%   Label 1 indicates wake
%   Label 0 indicates unconscious
% Then:
%   Label >= 1 indicates wake
%   Label <= 0 indicates unconscious
%   multidose: Isoflurane_1.2 = -1; Isoflurane_0.6 = 0; Wake = 1;
%       Post_Isoflurane = 2; Recovery = 3
%       Note - "Post_Isoflurane" condition not pre-registered
%   singledose: Isoflurane = 0; Wake = 1; PostIsoflurane = 2; Recovery = 3;
%       Note - "Recovery" condition not pre-registered
%           Not all flies have "Recovery" condition data
%   sleep: Inactive = 0; Active = 1

tic;

keywords = cellfun(@(x) strsplit(x, ','), data_labels, 'UniformOutput', 0);
condition_labels = cellfun(@(x) x{4}, keywords, 'UniformOutput', 0); % assumes condition labels are always the 4th keyword
condition_labels = cellfun(@(x) x(10:end), condition_labels, 'UniformOutput', 0); % remove leading "condition" (it's redundant)
labels = int8(nan(size(condition_labels)));

switch val_set
    case 'multidose'
        conditions = {...
            'Isoflurane_1.2',...
            'Isoflurane_0.6',...
            'Wake',...
            'Post_Isoflurane',...
            'Recovery'};
        condition_ids = [-1 0 1 2 3];
    case 'singledose'
        conditions = {...
            'Isoflurane',...
            'Wake',...
            'PostIsoflurane',...
            'Recovery'};
        condition_ids = [0 1 2 3];
    case 'sleep'
        conditions = {...
            'Inactive',...
            'Active'};
        condition_ids = [0 1];
end

for c = 1 : length(conditions)
    cmatch = cellfun(@(x) strcmp(x, conditions{c}), condition_labels);
    labels(cmatch) = condition_ids(c);
end

% Convert to binary labels (wake/unconscious) for scoring
labels_2classes = labels >= 1; % 1s and greater = wake

% Repeat labels to match prediction matrix trailing dimensions
pred_dims = size(predictions);
labels_2classes = repmat(labels_2classes, [1 pred_dims(2:end)]);

toc

%% Check accuracy per fly group

correct = cell(size(flyGroups));
accuracies = cell(size(flyGroups));
for g = 1 : length(flyGroups)
    
    correct{g} = predictions(flyGroup_rows{g}, :, :) == labels_2classes(flyGroup_rows{g}, :, :);
    accuracies{g} = squeeze(sum(correct{g}, 1) ./ size(correct{g}, 1))'; % (channels x features)
    
end

%% Check accuracy per class?
% e.g. to see accuracy of determining specifically post isoflurane
% (compared to wake)
% Note - important to look at accuracies per classes because number of
% wake/unconscious may not be balanced

tic;

correct_perCondition = cell(size(flyGroups));
accuracies_perCondition = cell(size(conditions));
for g = 1 : length(flyGroups)
    correct_perCondition{g} = cell(size(conditions));
    accuracies_perCondition{g} = cell(size(conditions));
    
    for c = 1 : length(conditions)
        condition_rows = find(labels == condition_ids(c) & flyGroup_rows{g});
        
        correct_perCondition{g}{c} = predictions(condition_rows, :, :) == labels_2classes(condition_rows);
        accuracies_perCondition{g}{c} = squeeze(sum(correct_perCondition{g}{c}, 1) ./ size(correct_perCondition{g}{c}, 1))'; % (channels x features)
        
    end
end

toc

%% Format to have same dimensions as training CV set?

% Format to save dims as training CV set
%   predictions - (ch x f x cv x cond x epoch)
%   labels - (ch x f x cv x cond x epoch)
%   accuracies - (ch x f x cv)

%% Save

tic;

group_names = {'multidose8', 'multidose4'};

for g = 1 : length(flyGroups)
    out_file = ['class_' class_type '_' group_names{g} val_suffix '_accuracy.mat'];
    
    s = struct();
    s.fly_group = flyGroups{g};
    s.fly_rows = flyGroup_rows{g};
    s.predictions = predictions(flyGroup_rows{g}, :, :);
    s.labels = labels(flyGroup_rows{g});
    s.labels_2classes = labels_2classes(flyGroup_rows{g}, :, :);
    s.correct = correct{g};
    s.correct_perCondition = correct_perCondition{g};
    s.accuracies = accuracies{g};
    s.accuracies_perCondition = accuracies_perCondition{g};
    s.conditions = conditions;
    s.condition_ids = condition_ids;
    
    save([out_dir out_file], '-struct', 's');
end

toc