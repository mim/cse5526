function p03wrapper()

trainFile = 'Z:\work\sync\5526\private\proj3data\proj3_train.lsv';
testFile = 'Z:\work\sync\5526\private\proj3data\proj3_test.lsv';

[y X] = libsvmread(trainFile);
%[X mu sig] = zscore(X);
model = svmtrain(y, X, '-t 0 -c 256 -g .125');
[pred acc probs] = svmpredict(y, X, model);
acc

[yte Xte] = libsvmread(testFile);
%Xte = bsxfun(@rdivide, bsxfun(@minus, Xte, mu), sig);
[pred acc probs] = svmpredict(yte, Xte, model);
