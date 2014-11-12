function L06wta(x)

if ~exist('x', 'var') || isempty(x), x = [.4 .2 .3 .4 .3]; end

N = length(x);
a = N/(N-1);
b = 1/(N-1);

printVec(0, 'y', x);
for i = 1:5
    xl = a * x - b * (sum(x) - x);
    x = lim(xl, 0, 1);
    printVec(i, 'yl', xl);
    printVec(i, 'y', x);
end


function printVec(i, label, vec)

fprintf('%s\tit=%d\t[', label, i);
fprintf('%g ', vec);
fprintf(']\n');
