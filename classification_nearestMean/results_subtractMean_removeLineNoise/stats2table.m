%% Description

%{

Convert stats.mat structure into a table

dataset; channel; analysis; condition-pair (conditionA; conditionB); performance; p; sig; valid

%}

clear stats

%% Load

source_file = 'stats.mat';

load(source_file);

%% Get list of fields in the structure

fields = fieldnames(stats);

for field = fieldnames
    
end