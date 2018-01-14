
data = dlmread('parzenData.txt', ' ');
size(data);
parzenWindowClassifier(data, [0.5 1.0 0.0; 0.31 1.51 -0.5; -0.3 0.44 -0.1], 1)


% *****************************************************
% Check the classification using squared distance
% A = [0.5 1 0; 0.31 1.51 -0.5; -0.3 0.44 -0.1];
% diff = 0;
% for j =1:3
%     disp('Character new');
%     diff = 0;
%     for i = 1:10
%         diff = diff+ sum((A(j,:) - data(i,1:3)).*(A(j,:) - data(i,1:3)));
%     end
%     diff
%     diff=0;
%     for i = 11:20
%         diff = diff+ sum((A(j,:) - data(i,1:3)).*(A(j,:) - data(i,1:3)));
%     end
%     diff
%     diff = 0;
%     for i = 21:30
%         diff = diff+ sum((A(j,:) - data(i,1:3)).*(A(j,:) - data(i,1:3)));
%     end
%     diff
% end

    