function [ L ] = log_probability(train_data_labeled, train_class_labeled, train_data_unlabeled, theta_w_log, theta_c_log)

% size of unlabeled training set
A_u = size(train_data_unlabeled, 1);

% number of different topics T
T = size(unique(train_class_labeled), 1);

%
D_l = size(train_data_labeled, 2);

%
D_u = size(train_data_unlabeled, 2);

% 
p_w_t_log = theta_w_log';

% initial log probability value
L = 0;

% evaluate the log probability P(A|theta) for the labeled data
for t = 1:T

    % calculate the class conditional probabilities P(A|T) 
    p_a_t_log = sum(bsxfun(@times, train_data_labeled, p_w_t_log(t, 1:D_l)), 2);

    % add the priot probabilities P(T) and update the log probability value
    L = L + sum(sum(bsxfun(@plus, p_a_t_log, theta_c_log(t))));
end

L;

% evaluate the log probability P(A|theta) for the unlabeled data. in this 
% case we'll do it a little bit differently, due to the 'log of sums' 
% element in expression 9 in Rodrigues2014.

p_t_a_u_log = zeros(A_u, T);

for t = 1:T

    % calculate the class conditional probabilities P(A|T) 
    p_a_t_log = sum(bsxfun(@times, train_data_unlabeled, p_w_t_log(t, 1:D_u)), 2);

    p_t_a_u_log(:, t) = p_a_t_log;

end

% add the prior probabilities P(T)
p_t_a_u_log = p_t_a_u_log + (ones(A_u, 1) * theta_c_log');

% run a 'log of sums' in a way not to incurr in floating point underflow
L = L + sum(log_sum(p_t_a_u_log, -200));

% add the P(theta) parcel
L = L + theta_prior(theta_w_log, theta_c_log, 2);

end

