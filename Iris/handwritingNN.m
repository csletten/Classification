%% Classify test data

close all

% Load MNIST data
load('handwritingData/data_all.mat');

slice_size = 1000;

% Classify test data and calculate errors
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

% Save test results to file
save('handwritingData/test_results.mat', 'test_results');

%% Plot

% Plot correctly and incorrectly classified images
correct_images_count = 2;
incorrect_images_count = 2;

for i=1:num_test
   match = test_results(i) == testlab(i);
   if (match && correct_images_count)
      figure
      image_matrix = zeros(row_size, col_size);
      image_matrix(:) = testv(i, :);
      image(image_matrix');
      title("Correctly classified as: " + string(testlab(i)));
      correct_images_count = correct_images_count -1;
   elseif (not(match) && incorrect_images_count)
      figure
      image_matrix = zeros(row_size, col_size);
      image_matrix(:) = testv(i, :);
      image(image_matrix');
      title("Classified as: " + string(test_results(i)) + ", correct classification: " + testlab(i));
      incorrect_images_count = incorrect_images_count -1;
   elseif (not(correct_images_count) && not(incorrect_images_count))
       disp(correct_images_count);
       disp(incorrect_images_count);
       break;
   end
end

% Plot confusion matrix
figure
cm = confusionchart(test_results, testlab);
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';