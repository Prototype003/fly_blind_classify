%% DESCRIPTION

%{

Converts  Fisher's Z to Pearson's r

Inputs:
    Fisher's z
Outputs:
    Pearson's r

%}

function [ r ] = fisher_zr( z )

r = (exp(2*z)-1) ./ (exp(2*z)+1);

end

