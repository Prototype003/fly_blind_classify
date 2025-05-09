%% Description

%{

Separate initialised HCTSA matrix (from TS_Init()) into separate files
    to falicitate parallel computation of HCTSA values per channel

Specifically for multidose dataset (due to large number of epochs)

%}

%% Settings

file_prefix = 'HCTSA_multidose'; % HCTSA_train; HCTSA_validate1; HCTSA_validate2
file_suffix = '.mat';

preprocess_string = '_subtractMean_removeLineNoise';

out_dir = ['hctsa_space' preprocess_string '/'];

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
