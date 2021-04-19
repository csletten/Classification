alpha = 0.01;
tol = 0.1;
labels = ["Setosa" "Versicolor" "Virginica"];
C = length(labels);
N = 90;
D = 4;

% Import data
x = zeros(D, N);

data1 = load('class_1.txt', '-ascii');
x(:, 1:30) = data1(1:30,:)'; 

data2 = load('class_2.txt', '-ascii');
x(:, 31:60) = data2(1:30,:)'; 

data3 = load('class_3.txt', '-ascii');
x(:, 61:90) = data3(1:30,:)'; 

% Generate t matrix
t = zeros(C, N);
for i=1:N
    if (i <= N/3)
        t(:, i) = [1 0 0]';
    elseif (i <= 2*N/3)
        t(:, i) = [0 1 0]';
    else 
        t(:, i) = [0 0 1]';
    end
    
end


W0 = randn(C, D);
b0 = randn(C, 1);
W = [W0 b0];

sigm = @(z) 1./(1+exp(-z));

while true
    grad_MSE = gradient(sigm, x, W, t, N);
    W = W - alpha*grad_MSE;
    
    if (norm(grad_MSE*alpha) < tol)
        break
    end
end

test_x = [6.1 2.9 4.7 1.4 1]';
disp(sigm(W*test_x));

function y = linearClassifier(sigm, x, W)
    y = sigm(W*[x' 1]');
end

function [grad_MSE, g] = gradient(sigm, x, W, t, N)
    grad_MSE = 0;
    for k = 1:N
        g = linearClassifier(sigm, x(:, k), W);
        grad_MSE = grad_MSE + ((g-t(:, k)).*g.*(1-g))*[x(:, k)' 1];
    end
end


