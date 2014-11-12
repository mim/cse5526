function [SSE MxAE grad] = p01mlpBackprop(W, x, d)

% Backprop for MSE for an MLP using sigmoid nonlinearity
%
% W is a cell array of weights, one for each layer, x is a matrix of data
% points, one point per column, d is a row vector of desired values.
% Returns the gradient of the MSE for each datapoint calculated using
% backprop.

% Forward pass
[y state] = p01mlpFwd(W, x);

% Final error(s)
e{length(state)} = d - y;
SSE = sum(0.5 * (d - y).^2, 2);
MxAE = max(abs(d - y), [], 2);

% Compute gradients and backpropagate errors
for layer=length(state):-1:2
    inputs = [state{layer-1}; ones(1,size(x,2))];
    phiPrime = state{layer} .* (1 - state{layer});
    delta{layer} = e{layer} .* phiPrime;
    
    eTmp = W{layer-1}' * delta{layer};
    e{layer-1} = eTmp(1:end-1,:);
    
    grad{layer-1} = delta{layer} * inputs';
end
