function [ids_valid] = getValidFeatures(TS_DataMat)
%GETVALIDFEATURES
%
% Exclusion criteria
%   Exclude any feature which has at least 1 NaN value across time series
%   Exclude any feature which has constant values across time series
%
% Inputs:
%   TS_DataMat = HCTSA data matrix (time series x features)
% Outputs:
%   ids_valid = logical vector; 1s indicate valid feature IDs and 0s
%       indicate feature IDs to exclude

% Get feature IDs with NaN
ids_nan = isnan(TS_DataMat);
ids_nan = any(ids_nan, 1);

% Get feature IDs with constant value
ids_const = diff(TS_DataMat, [], 1);
ids_const = all(~ids_const, 1);

ids_invalid = ids_nan | ids_const;

ids_valid = ~ids_invalid;

end

