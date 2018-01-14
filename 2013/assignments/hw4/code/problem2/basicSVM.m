function [test_data_Y] = basicSVM(train_data_X, train_data_Y, test_data_X, C)

    n = size(train_data_X, 1);
    
    % treshold for identifying lambda values which are 'larger than zero'
    % and not endup with too much support vectors (I've noticed that some
    % of those are 0.0000000...1, i.e. really small but > 0).
    treshold = 0.000001;

    % the class values in train_data_Y should be represented as -1 or +1
    % for direct analysis on an SVM, and note that train_data_Y is 
    % expressed as 0 or 1.
    y = 2.*train_data_Y - ones(n,1);
        
    % use quadprog() to solve the SVM quadratic problem (dual Lagrangian 
    % form). For that, one should use quadprog(H,f,[],[],Aeq,beq,lb,ub),
    % whith equality constraints, no inequalities and lower and upper 
    % bounds on the solutions.
    
    % calculate H (using the polynomial kernel (1 + x_i'x_j).^2).
    H = zeros(n,n);    
    H = (y*y').*((ones(n,1)*ones(n,1)' + train_data_X*train_data_X').^2);
    %H = (y*y').*((ones(n,1)*ones(n,1)' + train_data_X*train_data_X').^2 + (ones(n,1)*ones(n,1)'));
    
    % sum(lambda_n), for n = 1, ..., n. why the '-'? 
    f = -ones(n,1);

    % constraing sum(y_n.*lambda_n) = y'*lambda = 0.
    Aeq = y'; 
    beq = 0;
    
    % lower and upper bounds of lambda.
    lb = zeros(n,1);
    ub = C.*ones(n,1);
    
    % run quadprog note that the inequality restriction is not applied in
    % this case. Using the interior-point-convex Algorithm options yeilds
    % the best results.
    qp_opts = optimset('LargeScale','Off','Algorithm','interior-point-convex');
    lambda = quadprog(H,f,[],[],Aeq,beq,lb,ub,zeros(n,1),qp_opts);
    
    % indices of lambda for which lambda_n > some treshold, i.e. the
    % indices of the set S of support vectors.
    support_vectors_indeces = find(lambda > treshold);
    
    % indeces of data belonging to set M, having 0 < lambda < C. Since I've
    % imposed a treshold above, I'll do it again as it yeilds better
    % results than the < C bound.
    support_vectors_m_indeces = find(lambda > treshold & lambda < (C - treshold));
        
    % calculate b of the expression y = x'*w + b by averaging over all
    % support vectors correspondent with 0 < lambda < C.
    N_s = size(support_vectors_indeces,1);
    N_m = size(support_vectors_m_indeces,1);
    
    b = 0;
        
    for i=1:N_m
        for j=1:N_s
            b = b - lambda(support_vectors_indeces(j)).*y(support_vectors_indeces(j)).*(train_data_X(support_vectors_indeces(j),:)*train_data_X(support_vectors_m_indeces(i),:)' + 1).^2;
            %b = b - lambda(support_vectors_indeces(j)).*y(support_vectors_indeces(j)).*((train_data_X(support_vectors_indeces(j),:)*train_data_X(support_vectors_m_indeces(i),:)' + 1).^2 + 1);
        end
        
        b = b + y(support_vectors_m_indeces(i));
    end

    b = (1 / N_m) * b    
    
    % with all parameters calculated, i.e. with the training phase 
    % finished, one can now calculate the predicitions test_data_Y.
    N_y = size(test_data_X,1);
    test_data_Y = zeros(N_y,1);
    
    for i = 1:N_y
        for j = 1:N_s
            test_data_Y(i) = test_data_Y(i) + lambda(support_vectors_indeces(j)).*y(support_vectors_indeces(j)).*((test_data_X(i,:)*train_data_X(support_vectors_indeces(j),:)' + 1).^2);
            %test_data_Y(i) = test_data_Y(i) + lambda(support_vectors_indeces(j)).*y(support_vectors_indeces(j)).*((test_data_X(i,:)*train_data_X(support_vectors_indeces(j),:)' + 1).^2 + 1);
        end
        
        test_data_Y(i) = test_data_Y(i) + b;
        
        % make the class values as in the set {0;1} (remember we've been
        % working with the set {-1;1}).
        test_data_Y(i) = 0.5*((test_data_Y(i) / abs(test_data_Y(i)) + 1));
        
    end
    
return