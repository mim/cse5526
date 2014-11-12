function L03bTanh()

x = linspace(-5,5,101);
plot(x, [1.7159*tanh(2*x/3); 1.7159*2/3*sech(2*x/3).^2], 'LineWidth', 3)
grid
legend('\phi(v) = 1.7159 tanh(2x/3)', '\phi''(v) = 1.7159 2/3 sech^2(2x/3)', 'Location', 'SouthEast')
print('-dpng', 'Z:\work\sync\5526\private\pics\L03bTanh')
