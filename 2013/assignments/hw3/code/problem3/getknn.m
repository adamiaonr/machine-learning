function neighbors = getknn( training_data, test_point, k, mode)

N = size(training_data, 1);

% initialize array with test point's cartesian coordinates
%test_points = zeros(N, 2);
test_points = [test_point(:,1)*ones(1,N); test_point(:,2)*ones(1,N)]';
    
% calculate distances
%dist = zeros(N, 2);
dist = [sqrt((training_data(:,1) - test_points(:,1)).^2 + (training_data(:,2) - test_points(:,2)).^2), training_data(:,3)];

% sort dist from lowest to largest
dist = sortrows(dist);

% save the k first values of dist (do not return the test point itself)
% if mode is equal to 1, exclude the first point if dist = 0.
if (mode == 1)
    neighbors = dist(2:(k + 1),:);
else
    neighbors = dist(1:k,:);
end

end

