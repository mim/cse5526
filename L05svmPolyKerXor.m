function L05svmPolyKerXor()

x = linspace(-3,3,101);
y = linspace(-3,3,101);
[yg xg] = meshgrid(y, x);

colormap(easymap('rw', 3))
imagesc(x, y, xg.*yg < 0);
hold on
contour(x, y, xg.*yg, [0 0], 'b');
contour(x, y, xg.*yg, [-1 1], '--r');
plot([-1 1], [-1 1], 'ok', [-1 1], [1 -1], '*k')
hold off
xlabel('x_1')
ylabel('x_2')

print('-dpng', 'Z:\work\sync\5526\private\pics\L05_svmPolyKerXor');
