% test the built-in MNB classifier of the MATLAB statistic toolbox

% testing values
A_l = 2000;
A_l_t = 100;

A_test = 100;

% topic offset
offset = 0;

% the I array will be a mask which indicates which indices of train_data 
% will be used as labeled / unlabeled training datasets. 
I = zeros(A_l, 1);

% let's generate a balanced test_data dataset
Y = zeros(11269,1);

for t = 1:T
    
    % find the number of examples of articles of topic t.
    A_t = sum((train_class == t));
    
    t;
    
    index = randperm((A_t - A_test), A_l_t);
    index = bsxfun(@plus, offset, index);
    
    % set the computed indexes to 1 (0 are left for the unlabled data).
    I(index) = 1;    
        
    % set the indexes for the test data
    Y((offset + (A_t - A_test + 1)):(offset + A_t),:) = 1;
        
    % update the topic offset.
    offset = offset + A_t;
end

test_data_ = train_data((Y == 1),:);
test_class_ = train_class((Y == 1),:);

% train the classifier
mnb = NaiveBayes.fit(train_data((I == 1),:), train_class((I == 1),:), 'Distribution', 'mn');

% run the classifier over the 20 Newsgroups test set
bi_class = predict(mnb, test_data_);

a = ((bi_class - test_class_) == 0);
bi_accuracy = sum(a) / size(test_class_,1)

D = 53975;
alpha = 1;

% train a MNB text classifier
[theta_w, theta_c] = mnb_train(train_data((I == 1),:), train_class((I == 1),:), D, alpha, 'true');

% run the classifier on the test data
classification = mnb_class(log(theta_w), log(theta_c), test_data_, size(test_data_, 1));
%classification = mnb_class(log(theta_w), log(theta_c), test_data, size(test_data, 1));

% compute the fully-supervised accuracy
[maxs, index] = max(classification, [], 2);
super_accuracy = sum(((index - test_class_) == 0)) / (size(test_class_, 1))

diff = bi_accuracy - super_accuracy