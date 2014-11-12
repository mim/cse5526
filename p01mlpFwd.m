function [y state] = p01mlpFwd(W, x)

% Run forward pass of multi-layer perceptron
%
% W is a cell array of weight matrices, x is a matrix of data points, one
% point per column.

state{1} = x;
for layer = 1:length(W)
    stateAndBias = [state{layer}; ones(1,size(x,2))];
    state{layer+1} = sigmoid(W{layer} * stateAndBias);
end
y = state{end};


function y = sigmoid(v)
y = 1 ./ (1 + exp(-v));
