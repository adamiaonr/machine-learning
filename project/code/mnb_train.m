function [ theta_w, theta_c ] = mnb_train(train_data, train_class, dictionary_size, alpha, normalize)

% retrieve the number of different topics T
T = size(unique(train_class),1);

% size of the dictionary D, i.e. number of words
D = dictionary_size;

% range of words (contemplated in the dictionary) included in the train
% data
W = size(train_data,2);

% size (in articles) of the training dataset
A = size(train_data, 1);

% initialize the parameter matrices theta_w and theta_c
theta_w = zeros(D,T);
theta_c = zeros(T,1);

% calculate the theta_* matrices for every class in T
for t = 1:T
    
    % find the indices of all articles belonging to topic t
    ind = (train_class == t);
    
    % calculate the numerator of the expression 5 in Rodrigues2014 for all 
    % elements in theta_w
    theta_w(1:W, t) = theta_w(1:W, t) + sum(train_data(ind,:))';

    % the priors
    theta_c(t) = sum(train_class(ind,:) ./ t);

    % if the training requested normalization, normalize theta_w
    if strcmp(normalize,'true')

        theta_w(1:W,t) = theta_w(1:W, t) + alpha.*ones(W,1);

        % add the denominator and calculate the final theta_w
        theta_w(1:W,t) = theta_w(1:W,t) ./ (((alpha * W) + sum(sum(train_data(ind,:)))).*ones(W,1));
    end
    
end

% normalize P(T) array
if strcmp(normalize,'true')

%     theta_c = (alpha .* ones(T, 1)) + theta_c;
%     theta_c = theta_c ./ ((alpha * T) + sum(theta_c));

    %theta_c = (alpha .* ones(T, 1)) + theta_c;
    theta_c = theta_c ./ (sum(theta_c));

end

end

