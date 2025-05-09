%% DESCRIPTION

%{

Converts Pearson's r to Fisher's Z

Inputs:
    Pearson's r
Outputs:
    Fisher's z

%}

function [ z ] = fisher_rz(r)

z = (0.5) * log((1+r)./(1-r));

end

