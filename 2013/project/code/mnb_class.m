function [classification] = mnb_class(log_theta_w, log_theta_c, test_data, batch_size)

% number of articles in test data
A = size(test_data, 1);

% number of different topics for classification, T
T = size(log_theta_c, 1);

% size of the word dictionary within the test data.
D = size(test_data, 2);

% A x T matrix to hold the classification results, i.e. a row for each
% article and T columns, each with the posterior probability P(T|A). in
% fact this matrix will hold P(T)*P(A|T), the numerator of the Bayes Rule
% expression 7 on Rodrigues2014.
classification = zeros(A,T);

% pre-calculate the 1 x T matrix with the log(P(T))
L = log_theta_c';

% batch size
B = batch_size;

% pre-calculate the T x D matrix log(theta_w)
W = log_theta_w;
W = W';

% start the classification process, calculating the posterior probabilities
% in batches of B articles. one can not perform the classification over all
% documents in one batch, due to MATLAB memory constraints.

% initialize the batch limits b and b_end.
b = 1;

while b < A
    
    % update the final index of the batch. take action if we're in the last
    % batch and b_end isn't exactly b + B - 1.
    if (A - b >= B)
        b_end = b + B - 1;
    else
        b_end = A;
    end
        
    for t = 1:T
    
        % 1) calculate P(A|t) for the test articles, using expression 3 in
        % Rodrigues2014.
        t;
        
        % 1.1) prepare a B x D matrix to hold the theta_w parameters, for 
        % the current batch of B articles.
        %theta_w_exp = ones(b_end - b + 1,1) * (log(theta_w(:,t))');
        %theta_w_exp = ones(b_end - b + 1,1) * W(:,t)';

        % 1.2) we're now ready to perform the calculations of a B x D 
        % matrix, before we multiply its values over D. we do this in
        % logarithmic form (see expression x in Rodrigues2014), since the
        % multiplication of all class conditional probabilities P(w|t) may
        % result in floating point underflow.
        
        % 1.2.1) calculate the powers of P(w|t)^N, in logarithmic form
        class_cond = bsxfun(@times, test_data(b:b_end,:), W(t, 1:D));
        
        % 1.2.2) finalize the calculation of the class conditional P(A|t)
        classification(b:b_end, t) = sum(class_cond, 2);
    end
    
    % 1.2.3) the multinomial coefficients do not depend on the class, so
    % let's calculate them out of the 'class cycle', in logarithmic form
    %a_sizes = sum(test_data(b:b_end,:), 2);
    
    % 1.2.3.1) the factorials are calculated via the gamma function, i.e. 
    % gamma(n + 1) = n! the gammaln function calculates it in logarithmic
    % form
    %log_mn_coef = gammaln(a_sizes + 1) - sum(gammaln(test_data(b:b_end,:) + 1), 2);
    
    % 1.2.3.2) finally multiply (add in the log domain) the multinomial
    % coefficients to the classification matrix
    %classification(b:b_end,:) = bsxfun(@plus, classification(b:b_end,:), log_mn_coef);
    
    % 2) finalize the calculation of the posteriors P(T|A) by multiplying
    % P(A|T) by the priors P(T), bor the current batch of articles.
    classification(b:b_end,:) = classification(b:b_end,:) + (ones(b_end - b + 1,1) * L);
    
    % update the next batch starting point b.
    b = b_end + 1;
end

% % quicly calculate the accuracy (correct predictions) / (all predictions)
% [maxs, index] = max(classification, [], 2);
% 
% accuracy = sum(((index - test_class) == 0)) / (size(test_class, 1));
% accuracy

end

