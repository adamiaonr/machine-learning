function [ K D R ] = kmeans(data, k, initial)

% number of data points
N = size(data,1);

% array for the k-means k centers
K = initial;

% matrix of responsibilities R
R = zeros(N,k);

% matrix D to hold the squared distances between 
% dataset points and centers, save them in a N x k matrix. element D(i,j)
% holds the distance between data point i and cluster center j.
D = zeros(N,k);

% objective function (aka distortion measure) J
J = 1;
J_ = 0;

% iterative part of k-means, convergence is reached if objective function 
% J does not change.
while (J - J_) ~= 0

    % 1) E step of k-means

    % 1.1) find out the squared distances between data points and centers,
    % save them in D
    for i = 1:k

        % 1.1.1) 
        aux = (data - ones(N,1)*K(i,:));
        
        % 1.1.2) squared coordinates, [x^2 y^2] 
        aux = aux.*aux;

        % 1.1.3) squared distances to cluster center i: x^ + y^2
        D(:,i) = aux(:,1) + aux(:,2);
    end

    % 1.2) update the responsibility matrix R according to the min values 
    % in the matrix D
    [mins, index] = min(D, [], 2);
    
    R = zeros(N,k);

    for i = 1:k

        R(index == i,i) = 1;

    end
    
    % 2) M step of k-means, update parameters (centers of clusters)
    
    % 2.1) 1 / (SUM_n(R_nk))
    denom = diag((ones(k,1)./(R'*ones(N,1))));
    
    % 2.2) (1 / (SUM_n(R_nk))) * SUM_n(R_nk * x_n) and the final matrix of
    % cluster centers K
    K = denom*(R'*data);
    
    % 3) update the objective function J
    
    % 3.1) save the previous value of J
    J_ = J;
    
    J = ((R.*D)*ones(k,1))'*ones(N,1);

end

end

