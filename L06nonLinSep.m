function L06nonLinSep()

a = .5;
b = 1;
x = linspace(-1,1,101);
y = linspace(-1,1,101);
[yg xg] = meshgrid(y, x);

z = xg.^2/a^2 + yg.^2/b^2 < 1;
imagesc(x, y, z)

pts = [xg(:) yg(:)];

mapped = xg.^2/a^2 + yg.^2/b^2;
plot(mapped(:), z(:), '.')
