function Problem4_SGD(X,y)
% Example sample to use this function:
% X is a m * n matrix , we have m data point , the feature is n.
% y is the true value of the data point.
% X = [1 4 3;2 5 6 ;5 1 2 ;4 2 2];
% y = [19;26;19;20];
% subset must smaller than m , subset = 2;

[m,n] = size(X); % m is the number of dataset
alpha = 0.002;   %learning rate
num_iters = 200;
theta = zeros(n, 1);
subset = 1 % 1 is for SGD , b is for mini-batch, m is for GD

% cost function
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    index = randsample(length(X),subset,'false');
    subX = X(index', :);
    suby = y(index',:);
    
    h= zeros(subset,1);
    h = subX*theta;
    J_history(iter) = (1/(2*subset))*sum((h-suby).^2);
    
    tmp1 = zeros(size(subX,2),1);
    for i=1:subset
        tmp1= tmp1+(h(i)-suby(i)).*subX(i,:)';
    end
    
    theta = theta - (alpha/m)*tmp1;            %Every time to update the theta
    disp(J_history(iter));
    disp(theta);
end
end