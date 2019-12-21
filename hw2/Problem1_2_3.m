%Problem 1
%Problem 1.1
ytrclimateData = readtable('climate_change_1.csv');
train_data = climateData{climateData.Year <= 2006,:};
test_data = climateData{climateData.Year >2006,:};

ytrain = train_data(:,[11]);
xtrain = train_data(:,[3:10]);
Xtrain =[ones(length(xtrain),1) xtrain];

beta = closed_form_1(Xtrain,ytrain);

%Problem 1.2 Y=x* beta +epsilon
whichstats = {'beta','rsquare','tstat'};
Reg= regstats(ytrain,xtrain,'linear',whichstats);

ytest = test_data(:,11);
xtest = test_data(:,[3:10]);
Xtest =[ones(length(xtest),1) xtest];

yhat = Xtest * beta;
SSR = sum((yhat - mean(ytest)).^2);
SST = sum((ytest - mean(ytest)).^2);
R2Test = SSR/SST;

%Problem 1.3 the first 1 is for intercept
% 1 2 5 6 7 8
insignificantVariable = find(Reg.tstat.pval < 0.05);

%Problem 1.4 (x' * x)^ (-1) exists
climateData2 = readtable('climate_change_2.csv');
train_data2 = climateData2{climateData2.Year <= 2006,:};
test_data2 = climateData2{climateData2.Year >2006,:};

y2train = train_data2(:,12);
x2train = train_data2(:,[3:11]);
X2train =[ones(length(x2train),1) x2train];
beta2 = closed_form_1(X2train,y2train);
rank(X2train);

% Problem 2
% Problem 2.1 see in the homework.markdown

% Problem 2.2
climateData = readtable('climate_change_1.csv');
train_data = climateData{climateData.Year <= 2006,:};
test_data = climateData{climateData.Year >2006,:};

y = train_data(:,[11]);
x = train_data(:,[3:10]);
X =[ones(length(x),1) x];
beta_ridge = closed_form_2(X,y,10)

% Problem 2.3
beta_OLS = closed_form_1(X,y)
beta_ridge = closed_form_2(X,y,10)

% Problem 2.4
lambdaPara = [0.001,0.01,0.1,1,10];
y = train_data(:,[11]);
x = train_data(:,[3:10]);
X =[ones(length(x),1) x];

%Testing data R2
    yy = test_data(:,11);
    xx = test_data(:,[3:10]);
    XX =[ones(length(xx),1) xx];
    
for i =1:5
    beta_ridge = closed_form_2(X,y,lambdaPara(i));
    yhat = X * beta_ridge;
    SSR_train = sum((yhat - mean(y)).^2);
    SST_train = sum((y - mean(y)).^2);
    R2Test_train(i) = SSR_train/SST_train;
    
    yhat = XX * beta_ridge;
    SSR_test = sum((yhat - mean(yy)).^2);
    SST_test = sum((yy - mean(yy)).^2);
    R2Test_test(i) = SSR_test/SST_test;
end
disp(R2Test_train)
disp(R2Test_test)

% Problem 2.4 CV-validation in trainset
lambdaPara = [0.001,0.01,0.1,1,10];
[m,n] = size(train_data);
indices = crossvalind('Kfold',train_data(1:m,n),10);

for i = 1:5
    sumR2 = 0;
    for k = 1:10
        test = (indices ==k);
        train = ~test;
        train_data_new = train_data(train,:);
        test_data_new = train_data(test,:);
        y = train_data_new(:,11);
        x = train_data_new(:,3:10);
        X =[ones(length(x),1) x];
        beta_ridge = closed_form_2(X,y,lambdaPara(i));
        
        yy = test_data_new(:,11);
        xx = test_data_new(:,3:10);
        XX =[ones(length(xx),1) xx];
        
        yhat = XX * beta_ridge;
        ybar = mean(yy);
        SSE = sum((yy -  yhat).^2);
        meanMSE(i) = SSE/10
    end
end

%Problem 3
climateData = readtable('climate_change_1.csv');
train_data = climateData{climateData.Year <= 2006,:};
test_data = climateData{climateData.Year > 2006,:};

y = train_data(:,11);
x = train_data(:,3:10);
yy = test_data(:,11);
xx = test_data(:,3:10);

%X =[ones(length(x),1) x];
[B,FitInfo] = lasso(x,y,'CV',10);
fig = figure;
lassoPlot(B,FitInfo,'PlotType','CV')
legend('show');

idxLambda1SE = FitInfo.Index1SE;
coef = B(:,idxLambda1SE);
coef0 = FitInfo.Intercept(idxLambda1SE);

yhat_test = xx * coef + coef0;
%yhat_test = xx * coef;
SSR_test = sum((yhat_test - mean(yy)).^2);
SST_test = sum((yy - mean(yy)).^2);
R2Test = SSR_test / SST_test