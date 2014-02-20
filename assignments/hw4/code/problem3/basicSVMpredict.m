function [test_data_Y] = basicSVMpredict(lambda, b, y, x, test_data_X)    

    % find the relevant indeces for the input arrays
    idx = find(lambda > 0);
    
    % trim the input arrays
    lambda = lambda(idx);
    y = y(idx);
    x = x(idx,:);
    
    n = size(y,1);
        
    % the class values in train_data_Y should be represented as -1 or +1
    % for direct analysis on an SVM, and note that train_data_Y is 
    % expressed as e.g. 0 or 1.
    y_values = unique(y);
    y_values_diff = y_values(1) + y_values(2);
    y_values_div = abs(y_values(1) - (y_values_diff / 2));
    
    y = (y - (y_values_diff / 2).*ones(n,1)) ./ y_values_div;
    
    % with all parameters calculated, i.e. with the training phase 
    % finished, one can now calculate the predicitions test_data_Y.
    N_s = size(lambda, 1);
    N_y = size(test_data_X,1);
    test_data_Y = zeros(N_y,1);
    
    for i = 1:N_y
        for j = 1:N_s
            
            test_data_Y(i) = test_data_Y(i) + lambda(j).*y(j).*((test_data_X(i,:)*x(j,:)' + 1).^2);
        end
        
        test_data_Y(i) = test_data_Y(i) + b;
                
    end
        
    % apply the inverse transformation to y, so that the return values are 
    % not in the set {-1,1}
    test_data_Y = test_data_Y ./ abs(test_data_Y);
    test_data_Y = test_data_Y .* y_values_div;
    test_data_Y = test_data_Y + (y_values_diff / 2).*ones(N_y,1);
    
return