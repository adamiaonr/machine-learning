function [ L ] = log_sum(log_data, pw_limit)

    % sizes of log_data
    A = size(log_data, 1);
    T = size(log_data, 2);

    % retrieve the maximum values among each row of log_data
    m = max(log_data, [], 2);
    
    % subtract the values of m to each value on the respective row in log_data
    log_data = bsxfun(@minus, log_data, m);
    
    % find the positions of log_data with values larger than pw_limit        
    I = (log_data > pw_limit);
    
    % now let's briefly pass to the non-log domain for the 'log of sums' part
    data = zeros(A, T);

    data(I) = exp(log_data(I));
    data((I == 0)) = 0.0;
        
    % finally the result
    L = log(sum(data, 2)) + m;
    
end
