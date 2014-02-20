% this script provides the results for both k-means and EM algorithms, for
% 5 different initial means and multiple pairs {k, dataset}

close all

% 1) prepare the data for the algorithms

% 1.1) load the virus data points
load('joker.mat');

% 1.2) values of k
K = [4; 6];

% 2) call the cluster(k, dataset) function for eack {k,dataset} 
% combination.
for k = 1:size(K,1)
    
    % 2.1) each cluster() call generates 5 charts, for each {k,dataset}
    % combination. Each {k,dataset} shows an isolated grid of 2 x 5
    % subplots, top row with k-means results and lower row with EM for GMMs
    % results. 5 columns for each groups of initial means.
    cluster(K(k), joker1, 'joker1');
    cluster(K(k), joker2, 'joker2');    
end