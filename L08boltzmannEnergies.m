function [p E Z] = L08bolzmannEnergies(W, T, verbose)

% Compute the energy of all 2^D states of a Boltzmann machine with weights W

if ~exist('verbose', 'var') || isempty(verbose), verbose = false; end

D = size(W,1);
S = generateAllStates(D);

E = computeEnergy(W, S);
pt = exp(-1/T * E);
Z = sum(pt);
p = pt ./ Z;

if verbose
    printProbsAndStates(p, S);
end



function E = computeEnergy(W, X)
% Compute energy of a set of states
E = -.5 * sum(X .* (W * X), 1)';

function X = generateAllStates(D)
% Generate all 2^D states in {-1,1}^D
X = zeros(D, 2^D);
for i = 1:2^D
    bin = dec2bin(i-1);
    for b = 1:length(bin)
        X(length(bin)-b+1,i) = str2double(bin(b));
    end
    for b = length(bin)+1:D
        X(b,i) = 0;
    end
end

% Convert from {0,1} to {-1,1}
X = 2*X - 1;

function printProbsAndStates(p, S)
pm = '-+';
PM = pm((S+1)/2+1)';
for i = 1:length(p)
    fprintf('%d: %s %0.4f\n', i, PM(i,:), p(i));
end
