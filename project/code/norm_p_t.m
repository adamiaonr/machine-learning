function [ theta_c ] = norm_p_t(theta_c, alpha)

% retrieve the number of different topics T
T = size(theta_c, 1);

%theta_c = (alpha .* ones(T, 1)) + theta_c;
%theta_c = theta_c ./ ((alpha * T) + sum(theta_c));

theta_c = theta_c ./ (sum(theta_c));

end
