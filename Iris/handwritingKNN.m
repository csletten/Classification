load('handwritingData/data_all.mat');

K = 7;

class_count = 10;
M = 64;
data_per_class = num_train/class_count;
slice_size = 1000;

C = zeros(M*10, vec_size);

for i=1:class_count
    [~, C_i] = kmeans(trainv(trainlab==(i-1), :), M);
    C((M*(i-1)+1):(M*i), :) = C_i;
end

distances = dist(testv, C');

test_results = zeros(num_test, 1);
test_errors = 0;
for j=1:num_test
    [k_distance, k_indices] = mink(distances(j, :), K);
    for i=1:K
        k_indices(i) = floor((k_indices(i)-1)/M);
    end
    
    occurrences = histcounts(k_indices, 0:10);
    max_indices = find(occurrences==max(occurrences));
    
    if (length(max_indices) > 1)
        max_indices_distances = zeros(length(max_indices), 1);
        for i=1:length(max_indices)
            max_indices_distances(i) = sum(k_distance(k_indices == max_indices(i) - 1));
        end
        [~, min_dist_class] = min(max_indices_distances);
        disp(min_dist_class);
        
        test_results(j) = max_indices(min_dist_class) - 1;
    else
        test_results(j) = max_indices(1) - 1;    
    end
    
    if not(isequal(test_results(j), testlab(j)))
        test_errors = test_errors + 1;
    end
end

plotconfusion(test_results, testlab);