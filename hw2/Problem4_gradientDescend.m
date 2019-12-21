function Problem4_gradientDescend(X,y)
% Example sample to use this function:
% X is a m * n matrix , we have m data point , the feature is n.
% y is the true value of the data point.
% X = [1 4 3;2 5 6 ;5 1 2 ;4 2 2];
% y = [19;26;19;20];

[m,n] = size(X); % m is the number of dataset
alpha = 0.002;   %learning rate
num_iters = 1000;
theta = zeros(n, 1);

% cost function
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h= zeros(m,1);
    h = X*theta;
    J_history(iter) = (1/(2*m))*sum((h-y).^2);
    
    tmp1 = zeros(size(X,2),1);
    for i=1:m
        tmp1= tmp1+(h(i)-y(i)).*X(i,:)';
    end
    
    theta = theta - (alpha/m)*tmp1;            %Every time to update the theta
    disp(J_history(iter));
    disp(theta);
end
    plot(J_history)
    title('the cost function of 1000 iters')
end