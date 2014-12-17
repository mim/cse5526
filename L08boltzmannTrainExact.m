function [W ll allLike] = L08boltzmannTrainExact(X, nH, WMask, nIter, epsilon, T, wInit)

% Train a Boltzmann machine on data matrix X with nH hidden units
%
% W = L08boltzmannTrainExact(X, nH, WMask, nIter, eta, T, wInit)
%
% Training is done by exact marginalization of the terms in the likelihood
% gradient, so make sure to keep nV and nH small. X is an nV x K data
% matrix, nH is the number of hidden units to use, nIter is the number of
% iterations of gradient descent to use.  One visible unit will be appended
% that is always 1 to include the bias terms.

if ~exist('wInit', 'var'), wInit = []; end

[nV K] = size(X);
N = nV + 1 + nH;

XWithBias = [ones(1, K); X];
WMask = [0 ones(1,N-1); ones(N-1,1) WMask];

% posMat has each training state paired with all hidden states, grouped by
% training state
allHid = generateAllStates(nH);
posMat = zeros(N, K*2^nH);
for k = 1:K
    posMat(:,(k-1)*2^nH+1:k*2^nH) = [repmat(XWithBias(:,k), 1, 2^nH); allHid];
end

% negMat has all possible states
negMat = [ones(1, 2^(N-1)); generateAllStates(N-1)];

if isempty(wInit)
    W = 0.1*randn(N);
    W = W * W';
    W = W .* WMask;
else
% Trying out specific W structures for a specific problem...
%W = log(3) * [0 zeros(1,N-1); zeros(N-1,1) ones(N-1,N-1)-eye(N-1,N-1)];
%W = [0 0 0 0;
%    0 0 log(3) log(2);
%    0 log(3) 0 log(1);
%    0 log(2) log(1) 0];
    W = wInit;
end
ll = [];
for i = 1:nIter
    [rhoPlus pTilde] = expectedCorr(posMat, W, T, nH);
    [rhoMinus,pTildeAll,Z] = expectedCorr(negMat, W, T, N-1);
    rhoTotal = rhoPlus - K*rhoMinus;
    
    allLike(i,:) = pTildeAll / Z;
    like = pTilde / Z;
    ll(i) = sum(log(sum(reshape(like, 2^nH, K),1)));

    W = W + epsilon/T * rhoTotal .* WMask;
end
subplots({W(2:end,2:end), WMask, ll, allLike})


function [R pTilde Z] = expectedCorr(states, W, T, nMarg)
% Compute the probability of each state and the expected correlation
% between all pairs of neurons
pTilde = computeEnergy(W, states, T);

pTR = reshape(pTilde, 2^nMarg, []);
Z = sum(pTR,1);
pR = bsxfun(@rdivide, pTR, Z);
like = reshape(pR, 1, []);

R = states * bsxfun(@times, states, like)';


function [pTilde E] = computeEnergy(W, X, T)
% Compute unnormalized probability and energy of a set of states
E = -.5 * sum(X .* (W * X), 1);
pTilde = exp(- E / T);

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


function dW = numericalGradWrtW(W, posMat, negMat, T, nH, delta)
% Gradient checking code, should be that dW == (rhoPlus - K*rhoMinus)
N = size(W,1);
baseLL = logLike(W, posMat, negMat, T, nH);
for i = 1:N
    for j = 1:N
        WNew = W;
        WNew(i,j) = WNew(i,j) + 1e-4;
        WNew(j,i) = WNew(j,i) + 1e-4;
        newLL(i,j) = logLike(WNew, posMat, negMat, T, nH);
    end
end
dW = (newLL - baseLL) / delta;

function ll = logLike(W, posMat, negMat, T, nH)
% For numerical gradient
[rhoPlus pTilde] = expectedCorr(posMat, W, T, nH);
[rhoMinus,~,Z] = expectedCorr(negMat, W, T, size(W,1)-1);

like = pTilde/Z;
ll = sum(log(sum(reshape(like, 2^nH, []),1)));
