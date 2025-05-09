%% Description

%{

Quantify consistency of direction of effect of anesthesia, per fly

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['../hctsa_space' preprocess_string '/'];
data_source = 'multidose'; % multidose only

out_dir = ['results' preprocess_string '/'];

thresh_dir = ['results' preprocess_string '/'];
thresh_file = 'class_nearestMedian_thresholds';

addpath('../');
here = pwd;
cd('../'); add_toolbox; cd(here);

% Separate out first 8 and last 4 flies
flyGroups = {(1:8), (9:12)};
group_names = {'multidose8', 'multidose4'};

%% Load thresholds

load([thresh_dir thresh_file]);

%% Effect direction consistency

% Get data dimensions
[nChannels, nFlies, nConditions, nEpochs] = getDimensions(data_source);

% Get hctsa dimensions
tic;
tmp = load([source_dir 'HCTSA_' data_source '_channel1.mat']);
nRows = size(tmp.TS_DataMat, 1);
nFeatures = size(tmp.TS_DataMat, 2);
toc

switch data_source
    case 'train'
        conditions.conditions{1} = {'condition1'};
        conditions.cond_labels{1} = {'W'};
        conditions.conditions{2} = {'condition2'};
        conditions.cond_labels{2} = {'A'};
    case 'multidose'
        conditions.conditions{1} = {'conditionWake', 'conditionPost_Isoflurane', 'conditionRecovery'};
        conditions.cond_labels{1} = {'E1W', 'E1PostIso', 'E1R'};
        conditions.conditions{2} = {'conditionIsoflurane_0.6', 'conditionIsoflurane_1.2'};
        conditions.cond_labels{2} = {'E1A0.6', 'E1A1.2'};
    case 'singledose'
        conditions.conditions{1} = {'conditionWake', 'conditionPostIsoflurane'};
        conditions.cond_labels{1} = {'E2W', 'E2PostIso'};
        conditions.conditions{2} = {'conditionIsoflurane'};
        conditions.cond_labels{2} = {'E2A'};
    case 'sleep'
        conditions.conditions{1} = {'conditionActive'};
        conditions.cond_labels{1} = {'E3W'};
        conditions.conditions{2} = {'conditionInactive'};
        conditions.cond_labels{2} = {'E3S'};
end

% Get all pairings of the two condition classes
a = (1:length(conditions.conditions{1}));
b = (1:length(conditions.conditions{2}));
[A, B] = ndgrid(a, b);
pairs = [A(:), B(:)]; % each row gives the pair indexes

% Storage
consistencies = cell(size(pairs, 1), 1);
for p = 1 : size(pairs, 1)
    consistencies{p} = nan(nChannels, nFeatures, nFlies, nEpochs);
end

for ch = 1 : nChannels
    tic;
    
    % Load HCTSA values for the channel
    hctsa = load([source_dir 'HCTSA_' data_source '_channel' num2str(ch) '.mat']);
    
    % Get valid features
    %valid_features = getValidFeatures(hctsa.TS_DataMat);
    valid_features = ones(1, size(hctsa.TS_DataMat, 2)); % do for all features
    feature_ids = find(valid_features);
    
    for p = 1 : size(pairs, 1)
        
        % Get rows corresponding to each condition
        class1 = getIds(conditions.conditions{1}(pairs(p, 1)), hctsa.TimeSeries);
        class2 = getIds(conditions.conditions{2}(pairs(p, 2)), hctsa.TimeSeries);
        classes = {class1, class2};
        
        for fly = 1 : nFlies
            
            % Find rows corresponding to the fly
            fly_rows = getIds({['fly' num2str(fly)]}, hctsa.TimeSeries);
            
            % Get rows for each class
            rows = cell(size(classes));
            for class = 1 : length(classes)
                rows{class} = classes{class} & fly_rows;
            end
            
            for f = feature_ids
                
                % Get values for each class
                values = cell(size(classes));
                for class = 1 : length(classes)
                    values{class} = hctsa.TS_DataMat(rows{class}, f);
                end
                
                % Get direction of effect
                direction = directions(ch, f);
                
                % Flip epoch values to always test class1 > class2
                if direction == 0
                    values = cellfun(@(x) x*-1, values, 'UniformOutput', false);
                end
                
                for epoch = 1 : length(values{1})
                    
                    % Find proportion of class2 epochs which are in the same
                    % direction as the trained direction
                    greater = values{1}(epoch) > values{2};
                    
                    consistencies{p}(ch, f, fly, epoch) = sum(greater) / numel(greater);
                    
                end
                
            end
            
        end
        
    end
    t = toc;
    disp([data_source ' ch' num2str(ch) ' done in ' num2str(t) 's']);
end

%% Save

for g = 1 : length(flyGroups)
    tic;
    
    s = struct();
    s.conditions = conditions;
    s.pairs = pairs;
    s.consistencies = consistencies;
    for p = 1 : size(pairs, 1)
        % Restrict to the specific flies in the group
        s.consistencies{p} = s.consistencies{p}(:, :, flyGroups{g}, :);
    end
    
    out_file = ['consis_nearestMedian_' group_names{g}];
    save([out_dir out_file], '-struct', 's', '-v7.3');
    
    toc
end
