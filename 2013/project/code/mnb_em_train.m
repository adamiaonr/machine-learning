function [ theta_w, theta_c ] = mnb_em_train(train_data_labeled, train_class_labeled, train_data_unlabeled, train_class_unlabeled, dictionary_size, alpha, lambda)

% size of labeled training set
A_l = size(train_data_labeled,1);

% size of unlabeled training set
A_u = size(train_data_unlabeled,1);

% size of dictionary D
D = dictionary_size;

% partial size of the dictionary (that which is present in the training 
% data, as it may be different from D). we only include the total
% dictionary size D beforehand to produce theta_* parameters of the right
% dimensions for the testing phase. in our case, the knowledge of the full
% size of D, although incorrect, does not influence the calculations of
% theta_*.
W = size(train_data_unlabeled, 2);

% number of different topics T
T = size(unique(train_class_labeled),1);

% train a MNB classifier using the labeled data only, extract the theta_* 
% parameters obtained with the labeled data (do not normalize theta_*_l yet).
[theta_w_l, theta_c_l] = mnb_train(train_data_labeled, train_class_labeled, D, alpha, 'false');

% evaluate the complete log probability of the labeled data for the first
% time (use the normalized versions of theta_*_l now)
L = log_probability(train_data_labeled, train_class_labeled, train_data_unlabeled, log(norm_p_w_t(theta_w_l,alpha)), log(norm_p_t(theta_c_l,alpha)))

% create theta_* matrices for the parameter values trained with unlabeled 
% data (i.e. the overall model's theta_* parameters)
theta_w = theta_w_l;
theta_c = theta_c_l;

% confidence threshold
% p_confidence = 0.99;

% threshold for EM algorithm (0.05 as in Nigam2000)
threshold = 0.05;

% initialize delta_L, just so we're allowed to get into the cycle for at least 
% once
delta_L = 10 * threshold;

% max number of iterations
max_iter = 10;
n_iter = 0;

% EM algorithm (repeat while delta_L > threshold).
while ((delta_L > threshold) && (n_iter < max_iter))

    % 1) E-step 
    
    % 1.1) use the classifier with parameters log_theta_* to estimate the
    % posterior probabilities (responsibilities) of each article in the
    % unlabeled dataset.
    p_t_a_u_log = mnb_class(log(norm_p_w_t(theta_w(1:W,:),alpha)), log(norm_p_t(theta_c,alpha)), train_data_unlabeled, A_u);

    % 1.2) here we also employ a technique based on the log-sum-exp (LSE) 
    % technique.
    p_t_a_u = log_sum_exp(p_t_a_u_log, -200);
    
    % 1.3) Another intermidiate step, let's only select those values of 
    % P(T|A_u) for which we have large confidence, i.e. with P(T|A_u) >
    % p_confidence. the other values are reduced to 0.0.
    
%     I = (p_t_a_u > p_confidence);
%     p_t_a_u((I == 0)) = 0.0;
%     
%     for t = 1:T
%         
%         t
%         
%         I = (train_class_unlabeled == t);
%         
%         sum((train_class_unlabeled == t))
%         round(sum(p_t_a_u(I,t)))
%         round(sum(sum(p_t_a_u(I,:))))
%         
%     end
    
    % 1.4) just a check on EM's accuracy on unlabeled data. comment in future 
    % releases.
%     [maxs, index] = max(p_t_a_u, [], 2);
%     em_accuracy = sum(((index - train_class_unlabeled) == 0)) / (size(train_class_unlabeled, 1))
    
    % 2) M-step

    % 2.1) step of EM-lambda, reducing the influence of unlabeled data by a 
    % factor lambda, with 0 < lambda < 1
    p_t_a_u = bsxfun(@times, p_t_a_u, lambda);
    
    % 2.2) re-estimate the theta_* parameters. train using the unlabeled data 
    % only, add the values of theta_*_l (these wouldn't change anyway...) to 
    % the resulting theta_*_u.
    for t = 1:T

        % 2.2.1) note that we're handling a A(u) x D matrix here, hence this 
        % step will be computationally demanding.
        theta_w(1:W,t) = sum(bsxfun(@times, train_data_unlabeled, p_t_a_u(t)))';

        % 2.2.2) add the contribution of the theta parameters from the labeled 
        % data (in this case, theta_w_l can be directly added because we didn't 
        % normalize it).        
        theta_w(1:W,t) = theta_w(1:W,t) + theta_w_l(1:W,t);

        % 2.2.3) theta_c (priors P(T)), do not forget to add the labeled data
        % priors
        theta_c(t) = sum(p_t_a_u(t)) + theta_c_l(t);

    end
    
    % 3) Evaluate the complete log probability of the labeled and unlabeled
    % data and the prior P(theta_*) (see expression x in Rodrigues2014).

    % 3.1) save old value of L
    L_ = L;

    % 3.2) re-evaluate the complete log probability
    L = log_probability(train_data_labeled, train_class_labeled, train_data_unlabeled, log(norm_p_w_t(theta_w(1:W,:),alpha)), log(norm_p_t(theta_c,alpha)));

    % 3.3) compute delta_L
    delta_L = abs(L_ - L)
    
    % 3.4) Update number of iterations
    n_iter = n_iter + 1;

end

end

