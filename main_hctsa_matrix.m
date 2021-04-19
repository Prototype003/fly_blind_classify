%% Description

%{

% Plot feature matrix for given dataset

%}

%% Settings

source_dir = 'hctsa_space/';
source_file = 'HCTSA_train_channel6.mat'; % HCTSA_train.mat; HCTSA_validate1.mat;

%% Load

tic;
load([source_dir source_file]);
toc

%% Visualise

% Which set of time series to visualise for
keywords = {'fly1'};

% Get corresponding rows
match = true(size(TS_DataMat, 1), 1); keywords = {}; % everything
%match = getIds(keywords, TimeSeries);

% Get valid feature columns
valid_features = true(size(TS_DataMat, 2), 1); % everything
valid_features = getValidFeatures(TS_DataMat);

vis_rows = TS_DataMat(match, valid_features);

% Normalise
tic;
vis_rows = BF_NormalizeMatrix(vis_rows, 'mixedSigmoid');
toc

figure;
imagesc(vis_rows);
title([source_file(1:end-4) ' ' strjoin(keywords, ',')], 'Interpreter', 'none');

%% Manually add axis ticks to delineate groups
% Find a good way of doing this programmatically?

xlabel('feature');

% 13 flies x 8 epochs x 2 conditions
yticks((1 : 8 : 13*8*2));
ystrings = cell(size(yticks));
conds = {'W', 'A'};
y = 1;
for c = 1 : 2
    for f = 1 : 13
        ystrings{y} = ['F' num2str(f) ' ' conds{c}];
        y = y + 1;
    end
end
yticklabels(ystrings);
