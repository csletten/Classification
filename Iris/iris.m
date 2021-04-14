% Steepest descent variables
iterations = 100;
alpha = 0.1;

labels = ["Setosa" "Versicolor" "Virginica"];
C = length(labels);
global N
N = 90;
D = 4;


% Imported data
x = zeros(D, 90);

table1 = readtable('class_1.txt');
data1 = table1{:, :};
x(:, 1:30) = data1(1:30,:)'; 

table2 = readtable('class_2.txt');
data2 = table2{:, :};
x(:, 31:60) = data2(1:30,:)'; 

table3 = readtable('class_3.txt');
data3 = table3{:, :};
x(:, 61:90) = data3(1:30,:)'; 

global t
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

for i = 1:iterations
    direction = 0;
    
    W_grad = gradient(x, W);
    W = W - alpha*W_grad;
end

test_x = [5.1 3.8 1.6 0.2 1]';
disp(W*test_x);


function y = linearClassifier(x, W)
    sigm = @(z) 1./(1+exp(-z));
    y = sigm(W*[x' 1]');
end

function [W_grad, g] = gradient(x, W)
    global N
    global t
    W_grad = 0;
    for k = 1:N
        g = linearClassifier(x(:, k), W);
        W_grad = W_grad + ((g-t(:, k)).*g.*(1-g))*[x(:, k)' 1];
    end
end


