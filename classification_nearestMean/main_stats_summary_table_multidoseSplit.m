%% Description

%{

Treat multidose flies as two separate datasets (8 + 4 "odd" flies)

Table output

Show for every feature:
    validity
    nearestMedian performance + significance
    batchNormalised nearestMedian performance + significance
    consistency + significance

Fields:
    feature
    dataset
    condition pair
    performance
    significance
    significance threshold (before? after? FDR correction?)

Create tables for each dataset and performance type, then join?
Two outputs:
    One big joined table with everything
    One matfile with a struct of separate tables which can be joined
        together

%}

%%

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};

%%

tables = struct();

%% Load stats
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['results' preprocess_string filesep];
stats_file = 'stats_multidoseSplit.mat';

% Should have variable stats
load([stats_dir stats_file]);

%% Load hctsa files
% Assuming all hctsa files list all the features, we only need to
% load one file

ch = 1;

data_sources = {'train', 'multidose', 'singledose', 'sleep'};

source_dir = ['../hctsa_space' preprocess_string '/'];

hctsas = cell(size(data_sources));
for d = 1 : length(data_sources)
    
    source_file = ['HCTSA_' data_sources{d} '_channel' num2str(ch) '.mat'];
    
    disp(['loading ' source_file]);
    tic;
    hctsas{d} = load([source_dir source_file]);
    t = toc;
    disp(['loaded in ' num2str(t) 's']);
end

%% Load thresholds and directions

class_type = 'nearestMedian';
thresh_dir = ['results' preprocess_string '/'];
thresh_file = ['class_' class_type '_thresholds'];

load([thresh_dir thresh_file]);

%% Get a base table showing feature IDs and feature names
% Can get it from a hctsa file which gives all the features

tables.Operations = hctsas{1}.Operations;
tables.MasterOperations = hctsas{1}.Operations;

results = join(hctsas{1}.Operations, hctsas{1}.MasterOperations, 'LeftKeys', 'MasterID', 'RightKeys', 'ID');

nFeatures = size(tables.Operations, 1);

%% Iterate through each field in stats struct

% Base table - all tables to be joined to this one based on feature ID
results_table = tables.Operations;

% Field name format
%   performance type -> performance value -> dataset -> condition pair

% Get list of fields in stats
sfields = fieldnames(stats);
dset_labels = {'D', 'E1s8', 'E1s4', 'E2', 'E3'}; % 's' refers to 'splitting'

% Reverse labels so wake condition label comes first
%   Because the pairings in stats structure have the order unawake,wake
reverse_labels = [0 1 1 1 1];

% Iterate through each field in stats
for s = 1 : length(sfields)
    sfield = sfields{s};
    
    if length(sfield) > 14 && strcmp(sfield(end-14:end), 'BatchNormalised')
        dstring = sfields{s}(1:end-15);
        perf_types = {'nearestMedian'};
        perf_labels = {'class'};
        bn = 1; bn_string = 'Bn';
    else
        dstring = sfields{s};
        perf_types = {'nearestMedian', 'consis'};
        perf_labels = {'class', 'consis'};
        bn = 0; bn_string = '';
    end
    
    % Check which dataset the stats field corresponds to
    match = strcmp(dstring, dsets);
    d = find(match);
    
    [conds, cond_labels, cond_colours, cond_stats_order] = getConditions(dstring);
    conds = conds(cond_stats_order);
    cond_labels = cond_labels(cond_stats_order); % only gonna be using this
    cond_colours = cond_colours(cond_stats_order);
    
    % Replace special characters in cond_labels with something that can be
    % used in field names
    cond_labels = cellfun(@(x) strrep(x, '.', 'd'), cond_labels,...
        'UniformOutput', false);
    
    nearestMedian = table();
    
    for perf_counter = 1 : length(perf_types)
        perf_type = perf_types{perf_counter};
        
        channels_table = tables.Operations;
        
        % For each dataset + performance type, create a table with fields
        %   feature; validity, channel, sig_thresh, sig_thresh_fdr
        nChannels = size(stats.(sfield).(perf_type).sig, 1);
        for ch = 1 : nChannels
            id = tables.Operations.ID;
            channel = repmat(int8(ch), size(id));
            threshold = thresholds(ch, :)';
            direction = directions(ch, :)';
            
            tmp_fields = struct();
            tmp_tables = struct();
            
            % Following fields need to have a the dataset label appended
            valid = logical(stats.(sfield).valid_features(ch, :)');
            
            % Get performances from all condition pairings
            % perf needs to be replaced with the actual perf_type label
            tmp_fields.perf = cellfun(@(x) x(ch, :)',...
                stats.(sfield).(perf_type).performances,...
                'UniformOutput', false);
            tmp_fields.perf = cat(2, tmp_fields.perf{:});
            tmp_fields.sig = logical(permute(...
                stats.(sfield).(perf_type).sig(ch, :, :),...
                [2 3 1]));
            tmp_fields.p = permute(...
                stats.(sfield).(perf_type).ps(ch, :, :),...
                [2 3 1]);
            
            % Note - non-fdr values are constant across condition pairs
            %   But fdr values are are not (as they are determined
            %   based on the uncorrected significance values for the
            %   condition pair)
            tmp_fields.pThreshFdr = permute(...
                repmat(...
                stats.(sfield).(perf_type).ps_fdr(ch, :),...
                [1 1 nFeatures]),...
                [3 2 1]);
            tmp_fields.sigThresh = permute(...
                repmat(stats.(sfield).(perf_type).sig_thresh,...
                [1 nFeatures]),...
                [2 1]);
            tmp_fields.sigThreshFdr = permute(...
                repmat(...
                stats.(sfield).(perf_type).sig_thresh_fdr(ch, :),...
                [1 1 nFeatures]),...
                [3 2 1]);
            
            % Get/make field names for each condition pairing
            pairs = stats.(sfield).(perf_type).condition_pairs;
            nPairs = size(pairs, 1);
            pair_labels = cell(nPairs, 1);
            for pair = 1 : nPairs
                pair_labels{pair} = cond_labels(pairs(pair, :));
            end
            if reverse_labels(d) == 1
                pair_labels = cellfun(@fliplr, pair_labels,...
                    'UniformOutput', false);
            end
            % Prepend the dataset label to the pair
            pair_labels = cellfun(@(x) [dset_labels{d} '_' strjoin(x, 'x')],...
                pair_labels,...
                'UniformOutput', false);
            
            % Create table for fields which aren't repeated per
            %   condition pair)
            tmp = table(id, channel, threshold, direction, valid,...
                'VariableNames', {...
                'ID', 'channel', 'threshold', 'direction', ['valid_' dset_labels{d}]});
            
            % Create tables for repeating fields (repeats across condition
            %   pairs) and add condition pair labels
            fields = fieldnames(tmp_fields);
            for f = 1 : length(fields)
                
                if strcmp(fields{f}, 'perf')
                    % Replace 'perf' with the actual performance label
                    tmp_tables.(fields{f}) = array2table(...
                        tmp_fields.(fields{f}),...
                        'VariableNames',...
                        strcat([perf_labels{perf_counter} bn_string '_'], pair_labels));
                else
                    tmp_tables.(fields{f}) = array2table(...
                        tmp_fields.(fields{f}),...
                        'VariableNames',...
                        strcat([perf_labels{perf_counter} bn_string '_' fields{f} '_'], pair_labels));
                end
                
                % Join with the overall table
                tmp = [tmp tmp_tables.(fields{f})];
                
            end
            
            % Join across channels
            tmp = join(tables.Operations, tmp);
            if ch == 1
                channels_table = join(channels_table, tmp);
            else
                channels_table = [channels_table; tmp];
            end
            
        end
        
        % Matlab can't join tables when key variables have NaN values...
        %   note - numel(find(isinf(thresholds)) is 0, so can use inf as
        %   placeholder
        channels_table.threshold(isnan(channels_table.threshold)) = inf;
        
        if s == 1 && perf_counter == 1
            results_table = channels_table;
        else
            results_table = join(results_table, channels_table);
        end
        
    end
    
end

% Add back the NaNs (which were replaced with infs)
results_table.threshold(isinf(results_table.threshold)) = nan;

%% Export

out_file = 'results_table.xlsx';

disp(['writing table to ' out_file]);
tic;
writetable(results_table, out_file);
toc
disp([out_file ' saved']);
