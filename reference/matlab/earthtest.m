% Build command below should generate mex file earthmex
% mex -DSTANDALONE -DMATLAB -I../src earthmex.c ../src/earth.c ../src/blas/d_sign.c ../src/blas/daxpy.c ../src/blas/dcopy.c ../src/blas/ddot.c ../src/blas/dnrm2.c ../src/blas/dqrls.c ../src/blas/dqrsl.c ../src/blas/dscal.c ../src/blas/dtrsl.c ../src/R/dqrdc2.c

% ----------------------------------------------------------------------
% 1D approximation to sinusoid (example from earth.c main)
% ----------------------------------------------------------------------

% Create object
eobj1 = earth();

% Train
nCases = 100;
x = linspace(0,1,nCases)';  y = sin(4*x);
eobj1 = eobj1.train(x,y);

% Predict
np = 200;
x1 = linspace(min(x)-0.05,max(x)+0.05,np)';
y1 = eobj1.predict(x1);

% Display result
fh = figure;
plot(x1,y1,'g-'); 
hold on;  sh = scatter(x,y,'ro');  hold off;  axis tight;
title('Example 1');

% ----------------------------------------------------------------------
% Trees data from R
% Predict volume (column 3) from girth and height (columns 1:2)
% ----------------------------------------------------------------------

% Load data
trees = double(load('trees.txt'));

% Create object
eobj2 = earth();

% Train
eobj2 = eobj2.train(trees(:,1:2),trees(:,3));

% Predict on regular grid spanning training vectors
np = 50;
xgv = linspace(min(trees(:,1))-0.05,max(trees(:,1))+0.05,np);
ygv = linspace(min(trees(:,2))-0.05,max(trees(:,2))+0.05,np);
[Xm,Ym] = meshgrid(xgv,ygv);
zpr = eobj2.predict([Xm(:) Ym(:)]);
Zm = reshape(zpr,size(Xm));

% Display result
fh2 = figure;
mh = mesh(Xm,Ym,Zm);  
hold on;  sh = scatter3(trees(:,1),trees(:,2),trees(:,3),'ro');  hold off;
xlabel('girth');  ylabel('height');  zlabel('volume');
title('Example 2');