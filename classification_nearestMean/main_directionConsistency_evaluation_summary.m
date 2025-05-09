%% Description

%{

Quantify consistency of direction of effect of anesthesia, per fly

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['../hctsa_space' preprocess_string '/'];
data_sources = {'multidose', 'singledose', 'sleep'};

out_dir = ['results' preprocess_string '/'];

thresh_dir = ['results' preprocess_string '/'];
thresh_file = 'class_nearestMedian_thresholds';

addpath('../');
here = pwd;
cd('../'); add_toolbox; cd(here);

%% Load consistencies

out_prefix = 'consis_nearestMedian_';

cs = cell(length(data_sources), 1);
for d = 1 : length(data_sources)
    tic;
    cs{d} = load([out_dir out_prefix data_sources{d}]);
    t = toc;
    disp([data_sources{d} ' loaded in t=' num2str(t) 's']);
end

%% Plot distributions of consistency

ch = 6;

for d = 1 : length(data_sources)
    
    % Average across epochs (plot for each individual fly)
    nPairs = size(cs{d}.pairs, 1);
    nFlies = size(cs{d}.consistencies{1}, 3);
    figure;
    sp_counter = 1;
    for p = 1 : nPairs
        pair = cs{d}.pairs(p, :);
        
        tmp = mean(mean(cs{d}.consistencies{p}(:, :, :, :), 4), 3);
        
        subplot(1, nPairs, sp_counter);
        histogram(tmp(ch, :));
        title([cs{d}.conditions{d}.conditions{1}{pair(1)} ' x' newline cs{d}.conditions{d}.conditions{2}{pair(2)}], 'interpreter', 'none');
        xlabel('consistency');
        ylabel('feature count');
        xlim([0 1]);
        
        sp_counter = sp_counter + 1;
        
    end
    
end
