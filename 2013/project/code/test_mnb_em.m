% load the train and test datasets for the 20 Newsgroups detaset, available
% in http://qwone.com/~jason/20Newsgroups/
% i've made the dataset used in my MATLAB implementation in 
% http://paginas.fe.up.pt/~up200400437/dataset.zip

% create sparse matrices for the *.data datasets.
% train_data = spconvert(load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/train.data'));
% test_data = spconvert(load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/test.data'));
% 
% train_class = load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/train.label');
% test_class = load('/home/adamiaonr/Dropbox/Workbench/PhD/machine learning/project/experiments/dataset/test.label');

%train_data = spconvert(load('/home/adamiaonr/Documents/PhD/ml/train.data'));
%test_data = spconvert(load('/home/adamiaonr/Documents/PhD/ml/test.data'));

%train_class = load('/home/adamiaonr/Documents/PhD/ml/train.label');
%test_class = load('/home/adamiaonr/Documents/PhD/ml/test.label');

% 1) divide the training data into labeled and unlabeled training sets

% 1.1) problem variables:

% number of different newsgroups T (topics).
T = size(unique(train_class),1);

% size of dictionary D
D = max([size(train_data, 2); size(test_data, 2)]);

% size of training data
N = size(train_data, 1);

% 1.2) sizes of labeled data, based on the amount of labeled articles per
% newsgroup (must be less that 213 per each of the 20 newsgroups).
%A_l = [50 100 500 1000 2000 4000];
%A_u = [500, 1000, 3000, 5000, 7000, 8000, 9000];

% 1.2.1) testing values. comment in future releases.
A_l = 500;
A_u = N - A_l
A_l_t = A_l / T;

% topic offset
offset = 0;

% the I array will be a mask which indicates which indices of train_data 
% will be used as labeled / unlabeled training datasets. 
I = zeros(A_l + A_u, 1);

for t = 1:T
    
    % find the number of examples of articles of topic t.
    A_t = sum((train_class == t))
    
    index = randperm(A_t, A_l_t);
    index = bsxfun(@plus, offset, index);
    
    % set the computed indexes to 1 (0 are left for the unlabled data).
    I(index) = 1;    
    
    % update the topic offset.
    offset = offset + A_t;
end

% for t = 1:T
%    
%     sum((train_class((I == 0),:) == t))
%     
% end

% train the mnb_em classifier

% alpha for Laplace smoothing
alpha = 1;

% lambda for EM-lambda
lambda = 1.0;

% notice that the labeled dataset is based on the positions in the matrix I
% equal to 1, while the unlabeled dataset is formed by those positions
% equal to 0.

[theta_w, theta_c ] = mnb_em_train(train_data((I == 1),:), train_class((I == 1),:), train_data((I == 0),:), train_class((I == 0),:), D, alpha, lambda);
 
% test the classifier obtained after semi-supervised learning
semi_classification = mnb_class(log(norm_p_w_t(theta_w,alpha)), log(norm_p_t(theta_c,alpha)), test_data, size(test_data, 1));

% compute the semi-supervised accuracy
[maxs, index] = max(semi_classification, [], 2);
semi_accuracy = sum(((index - test_class) == 0)) / (size(test_class, 1))

% now train and run a normal mnb classifier, with the same training data
[theta_w_u, theta_c_u] = mnb_train(train_data((I == 1),:), train_class((I == 1),:), D, alpha, 'false');
classification = mnb_class(log(norm_p_w_t(theta_w_u,alpha)), log(norm_p_t(theta_c_u,alpha)), test_data, size(test_data, 1));

% compute the fully-supervised accuracy
[maxs, index] = max(classification, [], 2);
super_accuracy = sum(((index - test_class) == 0)) / (size(test_class, 1))

