function [ log_theta ] = theta_prior(log_theta_w, log_theta_c, alpha)

% calculation of the prior distribution over the parameters theta,
% represented by a Dirichlet distribution, as specified in Nigam2000. see
% expression x in Rodrigues2014.

% once again, due to danger of floating point underflow, we deal with
% logartithmic values instead of the actual theta_* values.
log_theta_w_sum = sum(((alpha - 1) .* log_theta_w));

log_theta = sum(((alpha - 1) .* log_theta_c') + log_theta_w_sum);

end

