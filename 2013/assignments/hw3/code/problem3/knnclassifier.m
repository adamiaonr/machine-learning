function prediction = knnclassifier(trainingData, testData, step, k)

M = 2;

N = size(trainingData, 1);
T = size(testData,1);

% initialize weights to 2
w = 2 .* ones(k + 1, 1);

% initialize exponential weight variables
exp_w = exp(w);
exp_w_sum = sum(exp_w);

% initialize gradient matrix
grad = zeros(k + 1, 1);

% initialize traning neighbor table
b = zeros(2, k + 1);
B = zeros(2, k + 1, N);

% initialize prediction [test point index, PC1|X (weight), PC2|X (weight), PC1|X, PC2|X]
prediction = zeros(T, 4);

% Build up neighbor table for each training point x
for t = 1:N
    
    % find k nearest neighbors of training point x, including their class C_k
    k_neighbors = getknn(trainingData, trainingData(t,1:2), k, 1);

    % build matrix B for k + 1 neighbors of training point x
    for i = 1:k

        if (k_neighbors(i,2) == 1)
            b(:,i) = [1; 0];
        else
            b(:,i) = [0; 1];
        end
    end

    b(:,k + 1) = [1/M; 1/M];
    
    B(:,:,t) = b;
end

stop = 0;

while stop == 0

    % building up the gradient matrix
    for t = 1:N

        % calculate SUM(B_uj * e^(wj)), for j = 1 ... k + 1
        B_exp_w_sum = B(trainingData(t,3),:,t) * exp_w;

        grad = grad + ((1 / B_exp_w_sum) * (B(trainingData(t,3),:,t) * diag(exp_w))'); 

    end

    grad = grad - (N / exp_w_sum) * exp_w;
    
    grad = grad .* step;

    % update the weights
    w = w + grad;
    
    % re-calculate the exponential weight matrix e^(wi)
    exp_w = exp(w);

    % re-calculate SUM(e^(wj)), for j = 1 ... k + 1
    exp_w_sum = sum(exp_w);    
    
    %(1 / exp_w_sum) .* exp_w 
    
    % stop when the change in weights is negligible
    % the submitted version went with an error: instead of max(abs(.)), it had 
    % abs(max(.)).
    if (max(abs(step .* grad)) < 0.001)
        
        stop = 1;
    end
    
end

% classify the testing data according to classes C_1 and C_2
for t = 1:T    
    
    % find k nearest neighbors of test point x, including their class C_k
    k_neighbors = getknn(trainingData, testData(t,1:2), k, 0);
    
    pc1_x = sum(k_neighbors(:,2) == 1) / k;
    pc2_x = sum(k_neighbors(:,2) == 2) / k;

    % build matrix B for k + 1 neighbors of training point x
    for i = 1:k

        if (k_neighbors(i,2) == 1)
            b(:,i) = [1; 0];
        else
            b(:,i) = [0; 1];
        end
    end

    b(:,k + 1) = [1/M; 1/M];
        
    PC1_X = (b(1,:) * exp_w) / exp_w_sum;
    PC2_X = (b(2,:) * exp_w) / exp_w_sum;
    
    prediction(t,:) = [PC1_X, PC2_X, pc1_x, pc2_x];
    
end

return
