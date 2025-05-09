%% Description

%{

Get chance accuracy distribution

%}

%% Settings

class_set = 'sleep_accuracy'; % multidose_accuracy; singledose_accuracy; sleep_accuracy; multidose8_accuracy; multidose4_accuracy

preprocess_string = '_subtractMean_removeLineNoise';

source_file = ['class_nearestMedian_' class_set '.mat'];
source_dir = ['results' preprocess_string '/'];

out_dir = source_dir;
out_file = ['class_random_' class_set '.mat'];

hctsa_prefix = '../hctsa_space/HCTSA_train';

%% Load

% Accuracies
acc = load([source_dir source_file]);

%% Chance distribution

% Convert correct labels to binary labels (wake/unconscious)
labels = acc.labels >= 1; % 1s and greater = wake

% Randomly predict for all epochs and features
nPredictions = numel(acc.predictions);
classes = unique(acc.predictions);
predictions = randsample(classes, nPredictions, true);
predictions = reshape(predictions, size(acc.predictions));

% Convert predictions to accuracies per condition pairing
nConditions = length(acc.condition_ids);
correct_perCondition = cell(1, nConditions);
accuracies_perCondition = cell(1, nConditions);
for c = 1 : nConditions
    
    % Get rows corresponding to the condition
    condition_rows = find(acc.labels == acc.condition_ids(c));
    
    % Check whether predictions are correct
    correct_perCondition{c} = predictions(condition_rows, :, :) == labels(condition_rows);
    
    % Get accuracy across epochs for the condition
    accuracies_perCondition{c} = squeeze(sum(correct_perCondition{c}, 1) ./ size(correct_perCondition{c}, 1))'; % (channels x features)

end

%% Save

save([out_dir out_file], 'predictions', 'correct_perCondition', 'accuracies_perCondition');