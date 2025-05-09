%% Description

%{

Get chance accuracy distribution

%}

%% Settings

class_set = 'sleep'; % multidose; singledose; sleep; multidose8; multidose4

preprocess_string = '_subtractMean_removeLineNoise';
source_file = ['consis_nearestMedian_' class_set '.mat'];
source_dir = ['results' preprocess_string '/'];

out_dir = source_dir;
out_file = ['consis_random_' class_set '.mat'];

hctsa_prefix = '../hctsa_space/HCTSA_train';

%% Load

% consistencies
con = load([source_dir source_file]);

%% Chance distribution

consistencies_random = cell(size(con.consistencies));
for pair = 1 : length(con.consistencies)
    dims = size(con.consistencies{pair});
    
    % Assumes equal number of epochs for each class in the condition pair
    pool = single((0:dims(4)) / dims(4));
    consistencies_random{pair} = randsample(pool, numel(con.consistencies{pair}), true);
    consistencies_random{pair} = reshape(consistencies_random{pair}, dims);
    
    % Keep only the first distribution (to save space)
    %   (use the same chance distribution for all channels)
    consistencies_random{pair} = consistencies_random{pair}(1, :, :, :);
end

%% Save

save([out_dir out_file], 'consistencies_random');