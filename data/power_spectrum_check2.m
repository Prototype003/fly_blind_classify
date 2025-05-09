%% Description

%{

Look at power spectrums in preprocessed data

%}

%%

clear tmp

%% Common parameters

sample_rate = 1000;

params = struct();
params.Fs = sample_rate;
params.tapers = [5 9];
params.pad = 2;
params.removeFreq = [];

data = struct();

% For keeping track if the datasets have been rereferenced
%   0 = not rereferenced; 1 = rereferenced
%   Position 1 is for single-dosage set; position 2 is for sleep set
reref_flags = nan(2, 1);

%% Load data - Multi-dosage flies

% Multi-dosage flies
%   Note - dataset has 2 files
preprocess_string = '_subtractMean_removeLineNoise';
source_dir = ['../hctsa_space' preprocess_string '/'];
source_file = 'multidose.mat';
tmp = load([source_dir source_file]);
data.multidose = tmp;

%% Compute power spectra (fft)

process_sets = {'multidose'};
chunkLength = 2250; % same duration epochs for all datasets

addpath('../');

for dset = length(process_sets) : -1 : 1
    tic;
    
    data.(process_sets{dset}).power = cell(size(data.(process_sets{dset}).labels));
    data.(process_sets{dset}).faxis = cell(size(data.(process_sets{dset}).labels));
    
    for chunk = 1 : length(data.(process_sets{dset}).timeSeriesData)
        
        Y = fft(data.(process_sets{dset}).timeSeriesData{chunk}(1:chunkLength));
        P2 = abs(Y/chunkLength);
        P1 = P2(1:chunkLength/2+1, :);
        P1(2:end-1, :) = 2*P1(2:end-1, :);
        f = params.Fs*(0:(chunkLength/2))/chunkLength;
        data.(process_sets{dset}).power{chunk} = P1;
        data.(process_sets{dset}).faxis{chunk} = f;
        
    end
    toc
end

%% Plot power spectra

ch = 6; % which channel to show for

process_sets = {'multidose', 'singledose', 'sleep'};
process_sets = {'multidose'};

for dset = 1 : length(process_sets)
    
    data.(process_sets{dset}).Keywords = data.(process_sets{dset}).keywords;
    
    [nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(process_sets{dset});
    conds = getConditions(process_sets{dset});
    
    ch_rows = getIds({['channel' num2str(ch)]}, data.(process_sets{dset}));
    %ch_rows = ones(size(data.(process_sets{dset}).keywords)); % if averaging across channels
    
    figure;
    
    for fly = 1 : nFlies
        fly_rows = getIds({['fly' num2str(fly)]}, data.(process_sets{dset}));
        
        subplot(floor(sqrt(nFlies)), ceil(sqrt(nFlies)), fly);
        hold on;
        
        for cond = 1 : nConditions
            cond_rows = getIds({conds{cond}}, data.(process_sets{dset}));
            
            cond_vals = data.(process_sets{dset}).power(ch_rows & fly_rows & cond_rows);
            
            % Average across epochs
            cond_vals = cat(2, cond_vals{:});
            cond_vals = mean(log(cond_vals), 2);
            
            plot(data.(process_sets{dset}).faxis{1}, cond_vals);
            
        end
        
        %ylim([-6 6]);
        ylabel('log(power)');
        xlim([min(data.(process_sets{dset}).faxis{1}) max(data.(process_sets{dset}).faxis{1})]);
        xlabel('f (Hz)');
        set(gca, 'xscale', 'log');
        title([process_sets{dset} ' fly' num2str(fly) ' ch' num2str(ch)]);
        
    end
end

%% Plot power spectra for multidose 8 + 4 flies

process_sets = {'multidose'};

dset = 1;
fly_colours = cat(1, repmat({[0 0 0 0.25]}, [8 1]), repmat({[1 0 0 0.25]}, [4 1]));

[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(process_sets{dset});
conds = getConditions(process_sets{dset});

figure;
set(gcf, 'Color', 'w');

for cond = 1 : length(conds)
    
    cond_rows = getIds({conds{cond}}, data.(process_sets{dset}));
    
    subplot(3, 2, cond); hold on
    
    legend_items = [];
    
    for fly = 1 : nFlies
        tic;
        
        fly_rows = getIds({['fly' num2str(fly)]}, data.(process_sets{dset}));
        
        % Get all rows for the condition and fly
        vals = data.(process_sets{dset}).power(cond_rows & fly_rows);
        
        % Average across epochs
        vals = cat(2, vals{:});
        vals = mean(log(vals), 2);
        
        % Plot
        p = plot(data.(process_sets{dset}).faxis{1}, vals, 'Color', fly_colours{fly});
        
        if fly == 1 || fly == 9
            legend_items = cat(1, legend_items, p);
        end
        
        toc
    end
    
    ylabel('log(power)');
    xlabel('f (Hz)');
    set(gca, 'xscale', 'log');
    %set(gca, 'yscale', 'log');
    title([process_sets{dset} ' ' conds{cond}], 'interpreter', 'none');
    axis tight
    legend(legend_items, 'MD8 fly', 'MD4 fly', 'Location', 'southwest');
    
end

%% Print figure

print_fig = 0;

if print_fig == 1
	
	axis tight
	box on
	
	figure_name = '../figures_stage2/fig_powerMD8vMD4';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
	
end