function [ w ] = ridgeRegression_func1( x, y, lam )  
    xTx = x'*x;  
    [m,n] = size(xTx);  
    temp = xTx + eye(m,n)*lam;  
    if det(temp) == 0  
        disp('This matrix is singular, cannot do inverse');  
    end  
    w = temp^(-1)*x'*y;  
end  