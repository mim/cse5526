function L02stochasticGradient()

% Illustrate stochastic gradient descent in one dimension. Model is a
% simple linear regression problem with mean square loss and several
% observations.

rng('default')
wt = 2.2;
bt = 0;
sig = 1;
N = 5;

e = sig .* randn(N,1);
x = linspace(1,3,N)';
ord = randperm(length(x));
x = x(ord);
d = x * wt + bt + e;

% Linear regression plot
plot(x, d, 'x', x, x*wt+bt, '--')
ax = axis();
axis(ax + 0.5*[-1 1 -1 1]);
xlabel('x')
ylabel('d')
print('-dpng', 'Z:\work\sync\5526\private\pics\L02stochasticGradientXspace')


% cost vs w plot
w = linspace(1,3,31);
plotWvsE(d, x, w);
print('-dpng', 'Z:\work\sync\5526\private\pics\L02stochasticGradientWspace')


% Stochastic gradient vs average
w0 = 1.4;
delta = 0.15;
[E0x E0 gradx grad] = updateWInfo(d, x, w0);
plotWvsEGrads(d, x, w, w0, delta, delta, E0x, E0, gradx, grad);
print('-dpng', 'Z:\work\sync\5526\private\pics\L02stochasticGradientWspaceWithGrad')

delta = 0.02;
plotWvsEGradsFinalPt(d, x, w, w0, delta, E0x, E0, gradx, grad);
print('-dpng', 'Z:\work\sync\5526\private\pics\L02stochasticGradientWspaceWithStep')

w00 = w0;
delta = 0.02;
for i = 1:3*N
    [E0x E0 gradx grad] = updateWInfo(d, x, w00);
    plotWvsEGradsFinalPt(d, x, w, w00, delta, E0x, E0, gradx, grad);
    print('-dpng', ['Z:\work\sync\5526\private\pics\' sprintf('L02stochasticGradientWspaceWithGradStep%02d', i)])
    %pause(.1)
    
    w00 = w00 + gradx(mod(i-1,N)+1)*delta;
end

w00 = w0;
delta = delta;
for i = 1:3
    [E0x E0 gradx grad] = updateWInfo(d, x, w00);
    plotWvsEGradsFinalPt(d, x, w, w00, delta, E0x, E0, gradx, grad);
    print('-dpng', ['Z:\work\sync\5526\private\pics\' sprintf('L02stochasticGradientWspaceWithAvgGradStep%02d', i)])
    pause(.1)
    
    w00 = w00 + grad*delta*N;
end


function [Ex E] = plotWvsE(d, x, w)
Ex = bsxfun(@minus, d, x * w).^2;
E = mean(Ex, 1);
plot(w, Ex, 'Color', .8*[1 1 1])
hold on
plot(w, E, 'k', 'LineWidth', 3)
hold off
ylabel('Square error')
xlabel('w')


function plotWvsEGrads(d, x, w, w0, deltax, delta, E0x, E0, gradx, grad)
plotWvsE(d, x, w);
hold on
%quiver(w0*ones(N,1), E0x, delta*ones(N,1), -gradx*delta, 0)
plot(w0, E0x, 'ro')
plot(w0, E0, '.r', 'MarkerSize', 30)
plot([w0+0*deltax w0+deltax]', [E0x E0x-gradx.*deltax]', 'r')
plot([w0 w0+delta]', [E0 E0-grad.*delta]', 'r', 'LineWidth', 3)
hold off

function plotWvsEGradsFinalPt(d, x, w, w0, delta, E0x, E0, gradx, grad)
plotWvsEGrads(d, x, w, w0, delta*gradx, delta*grad, E0x, E0, gradx, grad);

function [E0x E0 gradx grad] = updateWInfo(d, x, w0)
E0x = bsxfun(@minus, d, x * w0).^2;
E0 = mean(E0x, 1);
gradx = 2*bsxfun(@minus, d, x * w0) .* x;
grad = mean(gradx, 1);
