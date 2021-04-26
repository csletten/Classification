% Load data
load('handwritingData/data_all.mat');

slice_size = 1000;
test_errors = 0;

test_results = zeros(num_test, 1);
for i=1:(num_test/slice_size)
    distances = dist(testv(((i-1)*slice_size+1):i*slice_size, :), trainv');
    for j=1:slice_size
        [~, NN] = min(distances(j, :));
        disp(NN);
        test_results(j+(i-1)*slice_size) = trainlab(NN);
        if not(isequal(test_results(j+(i-1)*slice_size), testlab(j+(i-1)*slice_size)))
            test_errors = test_errors + 1;
        end
    end
end

save('test_results.mat', 'test_results');

%% Plot
plotconfusion(test_results, testlab)