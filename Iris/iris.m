clear W
%% Constants
alpha = 0.01;
tol = 0.001;
labels = ["Setosa" "Versicolor" "Virginica"];
C = length(labels);
N = 90;
N_test = 60;
D = 4;

%% Training
% Import data
x = zeros(D, N);

data1 = load('class_1.txt', '-ascii');
x(:, 1:30) = data1(1:30,:)'; 

data2 = load('class_2.txt', '-ascii');
x(:, 31:60) = data2(1:30,:)'; 

data3 = load('class_3.txt', '-ascii');
x(:, 61:90) = data3(1:30,:)'; 

% Generate t matrix
t1 = generate_solution(C, N);


W0 = randn(C, D);
b0 = randn(C, 1);
W = [W0 b0];

sigm = @(z) 1./(1+exp(-z));

while true
    grad_MSE = gradient(sigm, x, W, t1, N);
    W = W - alpha*grad_MSE;
    
    if (norm(grad_MSE*alpha) < tol)
        break
    end
end

%% Testing
test = ones(D, N_test);
test(:, 1:20) = data1(31:50,:)'; 
test(:, 21:40) = data2(31:50,:)'; 
test(:, 41:60) = data3(31:50,:)';

test_results = ones(C, N_test);

t2 = generate_solution(C, N_test);
errors = 0;

for i = 1:N_test
    result = sigm(W*[test(:, i); 1]);
    [~, class] = max(result);
    binary_result = [0 0 0]';
    binary_result(class) = 1;
    test_results(:, i) = binary_result;
    
    if not(isequal(test_results(:, i), t2(:, i)))
        disp("ERROR");
        errors = errors + 1;
    end
end

% Use plotConfusion() and histogram()

function y = linearClassifier(sigm, x, W)
    y = sigm(W*[x' 1]');
end

function [grad_MSE, g] = gradient(sigm, x, W, t1, N)
    grad_MSE = 0;
    for k = 1:N
        g = linearClassifier(sigm, x(:, k), W);
        grad_MSE = grad_MSE + ((g-t1(:, k)).*g.*(1-g))*[x(:, k)' 1];
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



