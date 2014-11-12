function [stableStates sourceStates trans E] = L07hopfieldAnalysis(W, async, maxIter)

% Find the stable states of a hopfield network with transition matrix W
%
% [stableStates sourceStates] = L07hopfieldAnalysis(W, async, maxIter)
%
% W is the transition matrix for the hopfield network being analyzed of
% dimension D x D.  stableStates is a D x K matrix of states that are
% stable under transformation phi(Wx).  sourceStates is a 1 x K cell array
% of D x Nk matrices, where all of the states represented by columns in the
% kth cell array transition eventually to the kth stable state.

if ~exist('async', 'var') || isempty(async), async = true; end
if ~exist('maxIter', 'var') || isempty(maxIter), maxIter = 22; end

D = size(W,1);
S = generateAllStates(D);

if async
    for b = 1:D
        trans(:,b) = statesToIds(phiAsync(W, S, b));
    end
else
    trans = statesToIds(phi(W, S))';
end

E = computeEnergy(W, S);
plotBiograph(trans, E, S);

X    = S;
Xp = zeros(size(X));
for i = 1:maxIter
    if all(Xp(:) == X(:))
        break
    end
    
    Xp = X;
    if async
        for b = 1:D
            X = phiAsync(W, X, b);
        end
    else
        X = phi(W, X);
    end
end
if i == maxIter,
    error('Did not converge, i = %d', i)
end

% Find mapping from start states to final states
xid = statesToIds(X);
sid = statesToIds(S);
[~,stableIds,finalIds] = unique(xid);
stableStates = X(:,stableIds);
for i = 1:max(finalIds)
    sourceStates{i} = S(:,finalIds == i);
end


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


function Xtp1 = phi(W, Xt)
% Compute step activation function of W*Xt, where 0 leads to keeping the
% same value as Xt. 
v = W * Xt;
Xtp1 = (v ~= 0) .* (2*(v > 0) - 1) + (v == 0) .* Xt;


function Xtp1 = phiAsync(W, Xt, bit)
% Compute step activation function of W*Xt, where 0 leads to keeping the
% same value as Xt. 
if nargin < 3, bit = randi(size(W,1)); end
v = W(bit,:) * Xt;
Xtp1 = Xt;
Xtp1(bit,:) = (v ~= 0) .* (2*(v > 0) - 1) + (v == 0) .* Xt(bit,:);


function ids = statesToIds(states)
% Convert a D x N matrix of {-1,1} states to a 1 x N vector of integers
ids = 1 + 2.^(0:size(states,1)-1) * (states > 0);

function E = computeEnergy(W, X)
% Compute energy of a set of states
E = -.5 * sum(X .* (W * X), 1)';

function plotBiograph(trans, E, S)

[s r] = size(trans);
T = sparse(repmat((1:s)',1,r), trans, ones([s r]), s, s);

pm = '-+';
PM = pm((S+1)/2+1)';
for i = 1:length(E)
    nodeNames{i} = sprintf('%d:%s E=%0.2f', i, PM(i,:), E(i));
end
bg = biograph(T, nodeNames, 'LayoutType', 'hierarchical');
view(bg)
