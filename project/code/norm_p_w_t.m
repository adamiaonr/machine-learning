function [ theta_w ] = norm_p_w_t(theta_w, alpha)

% retrieve the number of different topics T
T = size(theta_w, 2);

% size of the dictionary D, i.e. number of words
D = size(theta_w, 1);

% calculate the theta_* matrices for every class in T
for t = 1:T

    theta_w(:,t) = theta_w(:, t) + (alpha .* ones(D,1));

    % add the denominator and calculate the final theta_w
    theta_w(:,t) = theta_w(:,t) ./ (((alpha * D) + sum(theta_w(:,t))) .* ones(D,1));

end

end
