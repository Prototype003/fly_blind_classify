%%

%{

Compare Rhiannon's provided merge_table with the one I generated from "raw"
data

Rhiannon's provided merge_table - conditions are identical within fly
Expectation - the data from the first condition of each fly should be
    identical between Rhiannon's merge_table and the new merge_table

%}

%%

% Load old merge_data

% Load new merge_data

%% 

l = min([length(merge_table) length(merge_table_new)]);

%%

a_all = nan(numel(merge_table(1).pre_visual_lfp), l);
b_all = nan(size(a_all));
for row = 1 : l
    a_all(:, row) = merge_table(row).pre_visual_lfp(:);
    b_all(:, row) = merge_table_new(row).pre_visual_lfp(:);
end

%%

c = corr(a_all, b_all);

%%

figure;

imagesc(c); colorbar;