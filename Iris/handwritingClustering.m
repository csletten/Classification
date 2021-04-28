close all

% Load MNIST data
load('handwritingData/data_all.mat');

class_count = 10;
M = 64;
data_per_class = num_train/class_count;
slice_size = 1000;

% Clustering
C = zeros(M*10, vec_size);
for i=1:class_count
    [~, C_i] = kmeans(trainv(trainlab==(i-1), :), M);
    C((M*(i-1)+1):(M*i), :) = C_i;
end

% Calculate Eucledian distances from test data to templates
distances = dist(testv, C');

% Classify test data and calculate errors
test_results = zeros(num_test, 1);
test_errors = 0;
for j=1:num_test
    [~, pos]= min(distances(j, :));
    test_results(j) = floor((pos-1)/M);
    
    if not(isequal(test_results(j), testlab(j)))
        test_errors = test_errors + 1;
    end
end

% Plot confusion matrix
cm = confusionchart(test_results, testlab);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';