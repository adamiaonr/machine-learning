function [test_data_Y] = SVM_DAG(train_data_X, train_data_Y, test_data_X, Cvalue)

test_data_Y = zeros(size(test_data_X,1),1); 

N = size(train_data_X,1);

% being a DAG SVM, first one must train k(k - 1)/2 binary SVMs of the 
% type basicSVM(), for each combination of classes. I've altered the
% basicSVM() of problem 2 and added the functions basicSVMtrain() and
% basicSVMpredict().

% number of different classes in train_data_Y
n = size(unique(train_data_Y),1);

% get all pair combinations between classes in train_data_Y
comb = nchoosek(1:1:n,2);

% let's train the SVMs which will be part of the DAG
n_comb = size(comb,1);

% matrix which will hold the model parameters for each combination, i.e.
% each entry in the comb array. to fecth the parameters for SVM classifier
% (i,j), fectch models_v(:,:,i,j) and models_b(:,:,i,j).
models_v = zeros(N,4,n_comb,n_comb);
models_b = zeros(n_comb,n_comb);

for i = 1:n_comb
    
    % isolate the training data with the relevant classes, i.e. comb(i,1)
    % and comb(i,2).
    idx = find(train_data_Y == comb(i,1) | train_data_Y == comb(i,2));
    x = train_data_X(idx,:);
    y = train_data_Y(idx);
    
    % train the [comb(i,1), comb(i,2)] SVM.
    [lambda, b, y_, x_] = basicSVMtrain(x, y, Cvalue);
    
    N_s = size(lambda,1);
    
    % save the model parameters for each one of the combinations. note that
    % one pads the model arrays (e.g. lambda, y...) to have a dimension of
    % N, because although the number of support vector may vary from SVM to
    % SVM, if we want to keep a matrix with the model parameters, we need a
    % consistent dimension for all models. since the maximum of support
    % vectors is N, that's the appropriate limit.
    models_v(:,:,comb(i,1),comb(i,2)) = [padarray(lambda, [N - N_s 0], 'post'), padarray(y_, [N - N_s 0], 'post'), padarray(x_, [N - N_s 0], 'post')];
    models_b(comb(i,1),comb(i,2)) = b;
end

N_y = size(test_data_X,1);

% now to the DAG part... the idea is descend along a tree or DAG, each
% vertex of the tree being an SVM classifier. we start with a root node
% which evaluates classes k = 1 and l = 2. based on the output of the
% classifer, o, we then go to node k = o, l = l + 1. we do this
% successively unitl the leaf nodes are reached.
for i = 1:N_y

    k = 1;
    l = 2;
    
    % the classification process is finished when we reach the leaf nodes
    % level.
    while(l < (n + 1))
        
        % choose the SVM model (k,l).
        a = zeros(N,4);
        a = models_v(:,:,k,l);
        
        % predict the class of point test_data_X(i) by calling
        % basicSVMpredict with the model chosen above.
        c = 0;
        c = basicSVMpredict(a(:,1),models_b(k,l),a(:,2),a(:,3:4),test_data_X(i,:));
        
        % the index of the decision, based on the value of k, we choose the
        % appropriate SVM on the next level of the DAG (l + 1).
        k = c;
        
        % push towards the leafs.
        l = l + 1;
    end    
    
    % we have a final classification for test_data_X(i), save it on the
    % array.
    test_data_Y(i) = k;
end

return