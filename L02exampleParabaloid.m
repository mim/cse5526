function L02exampleParabaloid()

% Plot a 3D parabaloid with iso-contour lines

% Meaningful parameters
a = 2; b = 1; xc = -1; yc = -1; th = pi/4;

% Derived coefficients
A = a^2*sin(th)^2 + b^2*cos(th)^2; 
B = 2*(b^2-a^2)*sin(th)*cos(th); 
C = a^2*cos(th)^2 + b^2*sin(th)^2; 
D = -2*A*xc - B*yc; 
E = -B*xc-2*C*yc; 
F = A*xc^2+B*xc*yc+C*yc^2-a^2*b^2;

[y x] = meshgrid(linspace(-3,3,31), linspace(-3,3,31));
z = A*x.^2 + B*x.*y + C*y.^2 + D*x + E*y + F;

contour(x, y, z)
print('-dpng', 'Z:\work\sync\5526\private\pics\exampleParaboloidContour')

surf(x, y, z), shading interp, hold on, contour3(x,y,z,22,'k'), hold off
xlabel('X'), ylabel('Y'), zlabel('Z')
print('-dpng', 'Z:\work\sync\5526\private\pics\exampleParaboloid3d')

view(0, 90)
print('-dpng', 'Z:\work\sync\5526\private\pics\exampleParaboloidAbove')
