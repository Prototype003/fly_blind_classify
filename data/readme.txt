Pipeline
Bipolar rereference for single-dosage and sleep flies
Split into 2.25s epochs, update data format to match discovery dataset
Subtract mean per epoch
Remove line noise per epoch
Initialise hctsa matrices

Data
split2250_bipolarRerefType1_postPuffpreStim.mat - discovery flies (N = 13)
	bipolarRerefType1 - ch1 minus ch2, ch2 minus ch3, etc. (ch1 is most central channel)
delabelled_data.mat - pilot evaluation flies (N = 2)
	labels are in labelled/labelled_data_01.mat (variable: labelled_shuffled_data)
labelled_data_1_selected.mat - multi-dosage flies (N = 12; part 1)
labelled_data_2_selected.mat - multi-dosage flies (N = 12; part 2)
merge_table.mat - single-dosage flies (N = 18)
LFP_data.mat - sleep flies (N = 19)
