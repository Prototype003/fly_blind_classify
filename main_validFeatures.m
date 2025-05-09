%% Description

%{

Check how valid features extend through datasets

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

hctsa_dir = ['hctsa_space' preprocess_string '/'];

dataPrefixes = {'HCTSA_train', 'HCTSA_validate1'};

%% Load one dataset to get number of features and channels

hctsa = load([hctsa_dir dataPrefixes{1} '_channel1.mat']);
nFeatures = size(hctsa.TS_DataMat, 2);

tmp = load('data/preprocessed/fly_data_removeLineNoise.mat');
nChannels = size(tmp.data.train, 2);

%%

% Get valid features per channel
ch_valid_features = nan(nChannels, nFeatures, length(dataPrefixes));
ch_excluded = zeros(nChannels, 2, length(dataPrefixes)); % 2 exclusion stages

for d = 1 : length(dataPrefixes)
    
    for ch = 1 : nChannels
        tic;
        hctsa = load([hctsa_dir dataPrefixes{d} '_channel' num2str(ch) '.mat']);
        [valid_ids, valid] = getValidFeatures(hctsa.TS_DataMat);
        ch_valid_features(ch, :, d) = valid_ids; % store
        ch_excluded(ch, :, d) = valid;
        toc
    end
    
end

%% Check for features which are newly excluded in each following dataset

ch_newInvalid_features = zeros(nChannels, nFeatures, length(dataPrefixes));

for d = 2 : length(dataPrefixes)
    
    for ch = 1 : nChannels
        
        % Get all features which were previously valid
        valid_old = find(ch_valid_features(ch, :, d-1));
        
        % Get the validity status in the new dataset
        valid_new = ch_valid_features(ch, valid_old, d);
        
        % Find out if the features are now invalid
        newInvalid = ~valid_new;
        
        ch_newInvalid_features(ch, valid_old, d) = newInvalid;
        
    end
    
end

%% Number of new invalids per channel

d = 2;
mean(sum(ch_newInvalid_features(:, :, d), 2))

