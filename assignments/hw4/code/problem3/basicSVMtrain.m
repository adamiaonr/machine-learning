function [lambda, b, y, x] = basicSVMtrain(train_data_X, train_data_Y, C)

    n = size(train_data_X, 1);
    
    % treshold for identifying lambda values which are 'larger than zero'
    % and not endup with too much support vectors (I've noticed that some
    % of those are 0.0000000...1.
    treshold = 0.00000001;

    % the class values in train_data_Y should be represented as -1 or +1
    % for direct analysis on an SVM, and note that train_data_Y is 
    % expressed as e.g. 0 or 1.
    y_values = unique(train_data_Y);
    y_values_diff = y_values(1) + y_values(2);
    y_values_div = abs(y_values(1) - (y_values_diff / 2));
    
    y = (train_data_Y - (y_values_diff / 2).*ones(n,1)) ./ y_values_div;
            
    % use quadprog() to solve the SVM quadratic problem (dual Lagrangian 
    % form). For that, one should use quadprog(H,f,[],[],Aeq,beq,lb,ub),
    % whith equality constraints, no inequalities and lower and upper bonds
    % on the solutions.
    
    % calculate H (using the polynomial kernel (1 + x_i'x_j).^2).
    H = zeros(n,n);    
    H = (y*y').*((ones(n,1)*ones(n,1)' + train_data_X*train_data_X').^2);
    
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
        end
        
        b = b + y(support_vectors_m_indeces(i));
    end

    b = (1 / N_m) * b;    
    
    % prepare the outputs
    lambda = lambda(support_vectors_indeces);
    
    % apply the inverse transformation to y, so that the return values are 
    % not in the set {-1,1}
    y = y(support_vectors_indeces) ./ abs(y(support_vectors_indeces));
    y = y .* y_values_div;
    y = y + (y_values_diff / 2).*ones(N_s,1);
    
    x = train_data_X(support_vectors_indeces,:);
    
return