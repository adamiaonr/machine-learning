function [ U, E, M, Y ] = em(data, k, initial)

% threshold for log likelihood change
threshold = 0.0001;

% number of data points
N = size(data,1);

% dimension of data
d = size(data,2);

% initialize array for the k GMM means
U = initial;

% initialize array of k mixing coeficients M. initialize it so that each
% mixture component has equal mixing coefficients.
%p = randperm(N,k);
%M = (1/(ones(1,k)*p').*p)';
M = (1/k).*ones(k,1);

% initialize the k covariance matrices, with the inverse covariance
% matrices (precision matrices) proportional to the unit matrix as in 
% Figure 9.8 of Bishop2006 (in this case we use the unit matrix for all).
E = zeros(d,d,k);

for i = 1:k
    E(:,:,i) = eye(d);
end

% initialize the matrix of responsibilities Y
Y = zeros(N,k);

% initialize the matrix G to hold values for the GMM, for each
% value of k and all N data points
G = zeros(N,k);

% 0.1) initial value for the objective function (log likelihood) L
for i = 1:k
    
    % 0.1.1) distances between data points and initial means
    x_u_dist = (data - ones(N,1)*U(i,:));
    
    % 0.1.2) inverse of covariance matrices (precision matrices)
    E_ = (E(:,:,i)\eye(d));
    
    % 0.1.3) calculate the likelihood probability value of data points for 
    % each GMM component, save it in G
    G(:,i) = M(i)*(1/(2*pi))*(1/sqrt(det(E(:,:,i)))).*exp(-0.5.*(((x_u_dist*E_).*x_u_dist))*ones(d,1));

end

% 0.2) the actual objective function value (log likelihood) value L

% 0.2.1) the sum over all k within the logarithm term
SUM_G_K = G*ones(k,1);

L = log(SUM_G_K)'*ones(N,1);

% we're now entering the iterative part of the EM algorithm
L_ = 0;

% keep running the E and M steps until convergence (i.e. while 
% (\Delta)L < threshold)
while abs(L - L_) > threshold

    % 1) E step of the EM algorithm, get the values for the 
    % responsibilities Y. SUM_G_K has been calculated before and will be
    % re-calculated at the end of the cycle.
    Y = ((1./SUM_G_K)*ones(1,k)).*G;
    
    % 2) M step of the EM algorithm, re-estimation of the mixture's 
    % parameters U, E and M.

    % 2.1) N_k, sum of Y over N
    N_k = Y'*ones(N,1);

    % 2.2) U
    U = (1./N_k)*ones(1,d).*(Y'*data);
    
    % 2.3) M
    M = (1/N).*N_k;
    
    % 2.4) E for each k
    for i = 1:k

        % 2.4.1) calculate, for all N data points, (x_n - u_k), k equal 
        % to i.
        x_u_dist = (data - ones(N,1)*U(i,:));

        % 2.4.2) calculate a d x d matrix, proportional to the re-estimated 
        % E(:,:,i). 
        E(:,:,i) = x_u_dist'*((Y(:,i) * ones(1,d)) .* x_u_dist);

        % 2.4.3) the final touch to get E(:,:,i).
        E(:,:,i) = (1/N_k(i)).*E(:,:,i);

        % 2.4.4) update the matrix G
        G(:,i) = M(i)*(1/(2*pi))*(1/sqrt(det(E(:,:,i)))).*exp(-0.5.*(((x_u_dist*E_).*x_u_dist))*ones(d,1));

    end

    % 2.5) Evaluate the log likelihood

    % 2.5.1) save old value of L
    L_ = L;

    % 2.5.2) compute the new L
    SUM_G_K = G*ones(k,1);
    L = log(SUM_G_K)'*ones(N,1);

end
    
end
