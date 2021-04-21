close all
%% Constants
alpha = 0.01;
tol = 0.07;
labels = ["Setosa" "Versicolor" "Virginica"];
C = length(labels);
training_range = 1:30;
test_range = 31:50;
N_training = length(training_range)*3;
N_test = length(test_range)*3;
D = 4;
 
 
%% Training
% Import training data
x = zeros(D, N_training);
 
data1 = load('class_1.txt', '-ascii');
x(:, 1:30) = data1(training_range,:)';  
data2 = load('class_2.txt', '-ascii');
x(:, 31:60) = data2(training_range,:)'; 
data3 = load('class_3.txt', '-ascii');
x(:, 61:90) = data3(training_range,:)'; 
 
% Generate t matrix
t1 = generate_solution(C, N_training);
 
% Initialise W matrix 
W0 = rand(C, D);
b0 = rand(C, 1);
W = [W0 b0];
 
 
while true
   grad_MSE = gradient(x, W, t1, N_training, C, D);
   W = W - alpha*grad_MSE;
    
   if (norm(grad_MSE) < tol) 
       break
   end
end
 
training_results = ones(C, N_training);
training_errors = 0;
 
for i=1:N_training
    result = sigm(W*[x(:, i); 1]);
    [~, class] = max(result);
    binary_result = [0 0 0]';
    binary_result(class) = 1;
    training_results(:, i) = binary_result;
    if not(isequal(training_results(:, i), t1(:, i)))
        training_errors = training_errors + 1;
    end   
end
 
%% Testing
% Import test data
test = ones(D, N_test);
test(:, 1:20) = data1(test_range,:)'; 
test(:, 21:40) = data2(test_range,:)'; 
test(:, 41:60) = data3(test_range,:)';
 
test_results = ones(C, N_test);
 
t2 = generate_solution(C, N_test);
test_errors = 0;
 
for i = 1:N_test
    result = sigm(W*[test(:, i); 1]);
    [~, class] = max(result);
    binary_result = [0 0 0]';
    binary_result(class) = 1;
    test_results(:, i) = binary_result;
    
    if not(isequal(test_results(:, i), t2(:, i)))
        test_errors = test_errors + 1;
    end
end
 
figure
plotconfusion(training_results, t1);
title("Confusion matrix for training set");
figure
plotconfusion(test_results, t2);
title("Confusion matrix for test set");
 
function y = linearClassifier(x, W)
    y = sigm(W*[x' 1]');
end
 
function [grad_MSE, g] = gradient(x, W, t, N, C, D)
    grad_MSE = zeros(C, D + 1);
    for k = 1:N
        g = linearClassifier(x(:, k), W);
        grad_MSE = grad_MSE + ((g-t(:, k)).*g.*(1-g))*[x(:, k)' 1];
    end
end
 
function t = generate_solution(C, solution_length)
    t = zeros(C, solution_length);
    for i=1:solution_length
        if (i <= solution_length/3)
            t(:, i) = [1 0 0]';
        elseif (i <= 2*solution_length/3)
            t(:, i) = [0 1 0]';
        else 
            t(:, i) = [0 0 1]';
        end
    
    end
end
 
function g = sigm(z)
    g = [0 0 0]';
    for i=1:size(z)
        g(i) = 1/(1+expm(-z(i)));
    end
end