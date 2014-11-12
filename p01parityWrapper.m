function i = p01parityWrapper(nBits, nHid, nIter, eta, momentum, plotEvery)

if ~exist('plotEvery', 'var') || isempty(plotEvery), plotEvery = 1000; end

% Generate data
[x d] = generateData(nBits);

% Initialize network
rng(22)
W{1} = initLayer(nHid, nBits+1);
W{2} = initLayer(1, nHid+1);
lastUpdate = {0, 0};

% Pre-compute some variables for plotting
[testX1 testX2 gridX plotCols plotRows gridShape] = initPlotVars(nBits, nHid);

% Do training
for i = 1:nIter
    [sse(i) maxAe(i) grad] = p01mlpBackprop(W, x, d);
    for layer = 1:length(W)
        update = grad{layer} + momentum * lastUpdate{layer};
        W{layer} = W{layer} + eta * update;
        lastUpdate{layer} = update;
    end
    
    % Plotting
    if mod(i,plotEvery) == 0
        subplot(plotRows, plotCols, 1)
        plot(1:length(sse), sse, 1:length(maxAe), maxAe)
        xlabel('Iteration')
        ylabel('Error')
        legend('SSE', 'max(AE)')
        drawnow
        
        if nBits == 2
            subplot 232, plotOutput(W, gridX, testX1, testX2, gridShape, 'input', []);
            subplot 235, plotOutput(W, gridX, testX1, testX2, gridShape, 'input', [0 1]);
        end
        
        if nHid == 2
            subplot 233, plotOutput(W(2:end), gridX, testX1, testX2, gridShape, 'hidden', []);
            subplot 236, plotOutput(W(2:end), gridX, testX1, testX2, gridShape, 'hidden', [0 1]);
        end
    end
    
    % Stopping criterion
    if maxAe(i) < 0.05;
        break
    end
end


function W = initLayer(nOut, nIn)
W = 2 * rand(nOut, nIn) - 1;
%W = 1./sqrt(nIn) * randn(nOut, nIn);


function [x d] = generateData(nBits)
x = zeros(nBits, 2^nBits);
for i = 1:2^nBits
    bin = dec2bin(i-1);
    for b = 1:length(bin)
        x(length(bin)-b+1,i) = str2double(bin(b));
    end
    for b = length(bin)+1:nBits
        x(b,i) = 0;
    end
end
d = mod(sum(x,1),2);


function plotOutput(W, gridX, x, y, shape, axisName, ca)
outs = p01mlpFwd(W, gridX);
%imagesc(x, y, reshape(outs, shape));
surf(x, y, reshape(outs, shape));
view(0, 90)
shading interp
xlabel([axisName '0'])
ylabel([axisName '1'])
if ~isempty(ca), caxis(ca), end
%colorbar

function [testX1 testX2 gridX plotCols plotRows gridShape] = initPlotVars(nBits, nHid)
testX1 = linspace(0,1,21);
testX2 = linspace(0,1,21);
[gridX2 gridX1] = meshgrid(testX2, testX1);
gridX = [gridX1(:) gridX2(:)]';
gridShape = size(gridX1);
if (nBits == 2) || (nHid == 2)
    plotCols = 3;
    plotRows = 2;
else
    plotCols = 1;
    plotRows = 1;
end
