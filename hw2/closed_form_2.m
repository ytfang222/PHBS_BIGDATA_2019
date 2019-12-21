function beta = closed_form_2(x,y,lambda)
    [m,n] = size(x);
    I = eye(n);
    temp = x' * x + eye(n) *lambda;
    if det(temp) ==0
        disp('This matrix is singular, cannot do inverse');
    end
    beta = (temp)\(x'* y);
end
