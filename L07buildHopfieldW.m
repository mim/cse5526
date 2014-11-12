function W = L07buildHopfieldW(stableStates, noSelfConnect)

% Build a Hopfield network W matrix designed for stableStates to be stable
%
% W = L07buildHopfieldW(stableStates)
%
% stableStates is a D x K matrix where each column should be a stable state
% of the network.  W is a D x D matrix created using the Hebbian learning
% rule for Hopfield networks.

if ~exist('noSelfConnect', 'var') || isempty(noSelfConnect), noSelfConnect = true; end

[D K] = size(stableStates);
if noSelfConnect
    W = 1/D * (stableStates * stableStates' - K * eye(D));
else
    W = 1/D * (stableStates * stableStates');
end
