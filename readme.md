# Analysis code for registered report: Wakefulness can be distinguished from general anesthesia and sleep in flies using a massive library of univariate time series analyses
OSF project DOI 10.17605/OSF.IO/8WVSQ (stage 2)

## Data files

Data files available at https://osf.io/8wvsq/?view_only=8a056d1c573b4f23a6cf6cea8b976ddb
- Place `preprocessed/` (contains preprocessed data) in `data/`
- Place `hctsa_space_subtractMean_removeLineNoise/` (contains computed hctsa values) in this folder
- Place `results_subtractMean_removeLineNoise/` (contains classification results) in `classification_nearestMean/`

## Figure panel scripts

Modify parameters at top of scripts (and at lines indicate) as required for each dataset

### Figure 1 panels
- `main_hctsa_schematic2.m`
### Figure 2 panels
- `classification_nearestMean/main_summary_figure.m`
### Figure 3 panels
- `classification_nearestMean/main_summary_figure.m`
### Figure 4 panels
- `main_hctsa_matrix_evaluation_channelAverage.m`
- `classification_nearestMean/main_summary_figure_evaluation.m`
### Figure 5 panels
- `classification_nearestMean/main_features_timeSeries_example_evaluation_figure.m`
- `classification_nearestMean/main_feature_autoCorr_figure.m`
### Figure 6 panels
- `main_hctsa_matrix_evaluation_channelAverage.m` (modify Line 474)
- `main_featureCluster_allCh_evaluation.m` (modify Line 166)
### Figure 7 panels
- `main_hctsa_matrix_evaluation_dnv3.m`
- `classification_nearestMean/main_summary_figure_evaluation.m`
- `main_hctsa_matrix_evaluation_channelAverage.m` (modify Line 474)
- `main_featureCluster_allCh_evaluation.m` (modify Line 166)

### Supplementary Material S2

### Figure S1
- `main_hctsa_matrix_evaluation_channelOffset.m`
### Figure S2
- `data/power_spectrum_check2.m`
### Figure S3
- `data/PHDRJ001-Q1304/03_code/MATLAB/inspect_sleep_naotsugu_01.m`
- Note - this figure is generated using data prior to preprocessing (not provided due to size constraints)
### Figure S4
- `classification_nearestMean/main_feature_autoCorr_figure.m`

## Processing pipeline

Modify parameters at top of script as required for each dataset. Suffix `_evalution` in the script name generally means the script is for the evalaution flies

Computing hctsa values from preprocessed time-series:
1. Initialise hctsa files
	- `main_hctsa_1_init.m` (for stage 1 data)
	- `main_hctsa_1_init_evaluation.m` (for stage 2 data)
2. Compute hctsa values:
	- `main_hctsa_2_compute.m`;
	- `main_hctsa_2_compute_perChannel_multidose.m` (for multidose flies)
3. Separate hctsa files out per channel:
	- `main_hctsa_3_perChannel.m`

Classification:
1. Train classifiers on Stage 1 data
	- Cross-validation performan in discovery flies: `classification_nearestMean/main_nearestMean_crossValidation.m`
	- Train classifiers on all discovery flies: `classification_nearestMean/main_nearestMean_trainAll.m`
2. Evaluate classifiers on Stage 2 data
	- Evaluate classifier performance in evaluation flies:
		- `classification_nearestMean/main_nearestMean_validate.m`
		- `classification_nearestMean/main_nearestMean_validate_batchNormalise.m`
		- `classification_nearestMean/main_nearestMean_validate_batchNormalise_evaluation.m`
		- `classification_nearestMean/main_nearestMean_validate_accuracy_evaluation.m`
		- `classification_nearestMean/main_nearestMean_validate_accuracy_evaluation_multidoseSplit.m` (for MD8, MD4 flies)
	- Within-fly effect direction consistency:
		- `main_directionConsistency.m`
		- `main_directionConsistency_evaluation.m`
		- `main_directionConsistency_evaluation_multidoseSplit.m`
3. Statistical analyses
	- Random distributions
		- `chance_accuracy.m`
		- `chance_accuracy_evaluation.m`
		- `chance_consistency.m`
		- `chance_consistency_evaluation.m`
	- Statistical tests
		- `get_stats.m`
		- `get_stats_evalaution_multidoseSplit.m`
