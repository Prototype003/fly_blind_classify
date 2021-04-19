%% Description

%{

Exclude any feature which has at least 1 NaN value across time series
Exclude any feature which has a constant value across time series

Exclusion is done per channel

%}

%% Settings

file_prefix = 'HCTSA_train';
file_suffix = '.mat';

out_dir = 'hctsa_space/';

nChannels = 15; % is there an easy way to get this programmatically instead?

%% Separate into channels

% Filter out sub-file for each channel, containing only values for that
%   channel

% Separate out channels into separate HCTSA files
for ch = 1 : nChannels
    tic;
    ch_string = ['channel' num2str(ch)];
    ch_rows = TS_GetIDs(ch_string, [out_dir file_prefix file_suffix], 'ts');
    TS_FilterData(...
        [out_dir file_prefix file_suffix],...
        ch_rows,...
        [],...
        [out_dir file_prefix '_' ch_string file_suffix]);
    toc
end
