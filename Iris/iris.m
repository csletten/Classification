close all

alpha = 0.01;
tol = 0.07;
max_iterations = 100000;

classes = ["Setosa" "Versicolor" "Virginica"];
C = length(classes);
features = sort([1 2 3 4]);
D = length(features);

training_range = 21:50;
test_range = 1:20;
N_training = length(training_range)*3;
N_test = length(test_range)*3;

%% Training

% Load training data
training_data = zeros(D, N_training);
data1 = load('irisData/class_1.txt', '-ascii');
data2 = load('irisData/class_2.txt', '-ascii');
data3 = load('irisData/class_3.txt', '-ascii');

for i=1:length(features)
    feature = features(i);
    
    training_data(i, 1:30) = data1(training_range,feature)';
    training_data(i, 31:60) = data2(training_range,feature)';
    training_data(i, 61:90) = data3(training_range,feature)';
    
end

training_targets = generate_targets(C, N_training);

% Initialise W matrix
W0 = rand(C, D);
w0 = rand(C, 1);
W = [W0 w0];

iterations = 0;

% Calculate new gradient until the norm of the gradient is below the tolerance level
while true
    grad_MSE = gradient(training_data, W, training_targets, N_training, C, D);
    W = W - alpha*grad_MSE;
    iterations = iterations + 1;
    
    if (norm(grad_MSE) < tol || iterations >= max_iterations)
        break
    end
end


training_results = ones(C, N_training);
training_errors = 0;

% Calculate results and the errors using the training data
for i=1:N_training
    result = sigm(W*[training_data(:, i); 1]);
    [~, class] = max(result);
    binary_result = [0 0 0]';
    binary_result(class) = 1;
    training_results(:, i) = binary_result;
    
    if not(isequal(training_results(:, i), training_targets(:, i)))
        training_errors = training_errors + 1;
    end
end

%% Testing

% Load test data
test_data = ones(D, N_test);
for i=1:length(features)
    feature = features(i);
    
    test_data(i, 1:20) = data1(test_range,feature)';
    test_data(i, 21:40) = data2(test_range,feature)';
    test_data(i, 41:60) = data3(test_range,feature)';
    
end

test_targets = generate_targets(C, N_test);

test_results = ones(C, N_test);
test_errors = 0;

% Calculate results and the errors using the test data
for i = 1:N_test
    result = sigm(W*[test_data(:, i); 1]);
    [~, class] = max(result);
    binary_result = [0 0 0]';
    binary_result(class) = 1;
    test_results(:, i) = binary_result;
    
    if not(isequal(test_results(:, i), test_targets(:, i)))
        test_errors = test_errors + 1;
    end
end

% Plot confusion matrices
figure
plotconfusion(training_results, training_targets);
title("Confusion matrix for training set");
figure
plotconfusion(test_results, test_targets);
title("Confusion matrix for test set");

% Plot histograms
for i=1:length(features)
    feature = features(i);
    figure
    hold on
    histogram(data1(:, feature));
    histogram(data2(:, feature));
    histogram(data3(:, feature));
    hold off
    title("Feature " + string(feature));
    legend('Class 1', 'Class 2', 'Class 3');
end

% Calculates the discriminant
function y = discriminant(x, W)
y = sigm(W*[x' 1]');
end

% Calculate new gradient
function [grad_MSE, g] = gradient(x, W, t, N, C, D)
grad_MSE = zeros(C, D + 1);
for k = 1:N
    g = discriminant(x(:, k), W);
    grad_MSE = grad_MSE + ((g-t(:, k)).*g.*(1-g))*[x(:, k)' 1];
end
end

% Generate targets
function t = generate_targets(C, target_count)
t = zeros(C, target_count);
for i=1:target_count
    if (i <= target_count/3)
        t(:, i) = [1 0 0]';
    elseif (i <= 2*target_count/3)
        t(:, i) = [0 1 0]';
    else
        t(:, i) = [0 0 1]';
    end
    
end
end

% Sigmoid
function g = sigm(z)
g = [0 0 0]';
for i=1:size(z)
    g(i) = 1/(1+expm(-z(i)));
end
end