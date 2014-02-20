% this script provides the results for both k-means and EM algorithms, for
% 5 different initial means and one given pair {k, dataset}

close all

% 1) prepare the data for the algorithms

% 1.1) load the virus data points
load('joker.mat');

% 1.1.1) print the datasets to answer question 1.1
colors = ['c', 'g', 'm', 'b', 'y', 'r'];

figure(1);

hold on;

subplot(1, 2, 1);

xlabel('x_1');
ylabel('x_2');

plot(joker1(:,1), joker1(:,2), sprintf('%so', colors(6)), 'MarkerSize', 5, 'MarkerEdgeColor', colors(6), 'MarkerFaceColor', colors(6));

title('Outbreak 1');

subplot(1, 2, 2);

xlabel('x_1');
ylabel('x_2');

plot(joker2(:,1), joker2(:,2), sprintf('%so', colors(4)), 'MarkerSize', 5, 'MarkerEdgeColor', colors(4), 'MarkerFaceColor', colors(4));

title('Outbreak 2');

hold off;

% 1.2) total number of data points (both datasets have the same size)
N = size(joker1, 1);

% 1.3) CHOOSE THE DATASET HERE!!!
data = joker1;

% 1.3.1) just an experiment, maintain commented
%data = data(randperm(N,50),:);

% 1.4) CHOOSE THE NUMBER OF CLUSTERS HERE!!!
k = 4;

% 2) k-means algorithm

% 2.1) determine k initial centers for k-means. for now will be 
% positioned over k random data points.
p = randperm(size(data, 1),k);
initial = data(p,:);

% 2.2) determine the k-means cluster centers, distances of points 
% to the centers and responsibility matrix
[centers, D, R] = kmeans(data, k, initial);

% 2.3) plot the data points for k-means, use different colors for each 
% cluster, add the following to the next row of the subplot (k-means 
% and EM will be side-by-side)

figure(2);

subplot(1, 2, 1);

hold on;

xlabel('x_1');
ylabel('x_2');

% 2.4) the matrix of responsibilities R is used to determine the
% allocation of datapoints to their respective clusters k.
for i = 1:k

    outbreak = plot(data(R(:,i) == 1, 1), data(R(:,i) == 1, 2), sprintf('%so', colors(i)), 'MarkerSize', 5, 'MarkerEdgeColor', colors(i), 'MarkerFaceColor', colors(i));

end

% 2.5) plot the initial and final centers of the clusters found by 
% k-means.

for i = 1:k

    plot(initial(i, 1), initial(i, 2), 'wo', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w');
    init = plot(initial(i, 1), initial(i, 2), sprintf('%s*', colors(i)), 'MarkerSize', 10, 'LineWidth', 3);

    plot(centers(i, 1), centers(i, 2), 'wo', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w');    
    final = plot(centers(i, 1), centers(i, 2), sprintf('%s+', colors(i)), 'MarkerSize', 10, 'LineWidth', 3);
end

legend([outbreak init final], 'Outbreak', 'K-Means centers (init)', 'K-Means centers (final)');

title(sprintf('K-Means (k = %d)',k));

hold off;

% 3) EM algorithm for GMMs (Bishop2006, section 9.2.2) using the 
% centers provided by k-means to initialize the Gaussin Mixture Model 
% (initial values for the means). the returned values are: 
%   -# U matrix with means for each GMM component k; 
%   -# E covariance matrices for each GMM component k;
%   -# M matrix with mixing coefificents for each GMM component k;
%   -# Y, N x k matrix of responsibilities, for each data point and 
%       component k    
[U, E, M, Y] = em(data, k, centers);

% 3.1) plot the data points, with different k colours according to the
% cluster each point belongs to.

subplot(1, 2, 2);

hold on;

xlabel('x_1');
ylabel('x_2');

% 3.2) get the values of k (columns of Y) for which the 
% responsibilities are maximum, save them in I. based on these, the
% points are 'classified' as belonging to cluster k.
[C, I] = max(Y');

for i = 1:k

    % if any of the GMM components doesn't contribute with enough
    % responsibility, don't plot it.
    if sum(I' == i) > 0
        outbreak = plot(data(I' == i,1), data(I' == i,2), sprintf('%so', colors(i)), 'MarkerSize', 5, 'MarkerEdgeColor', colors(i), 'MarkerFaceColor', colors(i));
    end
end

% 3.3) plot the ellipse contours of the Gaussian mixture model obtained 
% via EM (code based on a tutorial on
% http://www.cs.columbia.edu/~jebara/6998-01/plotGauss.m).
% In addition, plot each group k of data points in different colours,
% according to EM classification.
for i = 1:k

    if sum(I' == i) > 0
        t = -pi:.01:pi;
        x = sin(t);
        y = cos(t);

        [vv, dd] = eig(E(:,:,i));
        A = real((vv * sqrt(dd))');
        z = [x' y'] * A;

        plot(z(:,1) + U(i,1), z(:,2) + U(i,2), 'w', 'LineWidth', 6);
        contour = plot(z(:,1) + U(i,1), z(:,2) + U(i,2), colors(i), 'LineWidth', 3);
    end
end

% 3.4) plot the initial means (final k-means centers) and final means for
% the EM algorithm.
for i = 1:k

    if sum(I' == i) > 0
        % 3.4.1) draw the k-means final centers
        plot(centers(i,1), centers(i,2), 'wo', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w');    
        final = plot(centers(i,1), centers(i,2), sprintf('%s+', colors(i)), 'MarkerSize', 10, 'LineWidth', 3);

        % 3.4.2) plot the means of the Gaussian mixture model obtained via EM
        plot(U(i,1), U(i,2), 'wo', 'MarkerSize', 15, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w');
        em_ = plot(U(i,1), U(i,2), sprintf('%sx', colors(i)), 'MarkerSize', 10, 'LineWidth', 3);
    end
end

legend([outbreak contour final em_], 'Outbreak', 'Gaussian variance contours', 'K-Means centers (final)', 'EM centers');

title(sprintf('GMM after EM (k = %d)',k));

hold off;
