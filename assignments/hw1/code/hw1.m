clear

% load the problem's data
% change the argument of load() to the location of trainingset.mat in your 
% system
load('../trainingset.mat');

% total number of observations from the training sets
N = length(X1(:,1)) + length(X2(:,1)) + length(X3(:,1));

% calculate the parameters for each one of the distributions Xk, to be 
% used in the discriminant functions dscrmnt(x, ...)

% mean
U1 = mean(X1);

% covariance matrix
Cov1 = cov(X1);

% inverse of the covariance matrix (let's calculate it beforehand and not 
% every time dsrmnt(x, ...) is ran)
iCov1 = inv(Cov1);

% the factor which affects the Gaussian distributions of Xk, including 
% the prior probability for each class, pri
pr1 = length(X1(:,1))/(N);
k1 = pr1/(2*pi*sqrt(det(Cov1)));

U2 = mean(X2);
Cov2 = cov(X2);
iCov2 = inv(Cov2);
pr2 = length(X2(:,1))/(N);
k2 = pr2/(2*pi*sqrt(det(Cov2)));

U3 = mean(X3);
Cov3 = cov(X3);
iCov3 = inv(Cov3);
pr3 = length(X3(:,1))/(N);
k3 = pr3/(2*pi*sqrt(det(Cov3)));

% for convenience, create arrays to accomodate the each observation 
% belonging to class Ck
C1 = [];
C2 = [];
C3 = [];

figure;
hold on;

[r,c] = size(XX);

% for each observation x in XX, determine its class Ck following the 
% heuristics defined in 2.3 b)
for i=1:r

    x = XX(i,:);
    
    % descriminant function template for C1 and C2
    if dscrmnt(x,U1,U2,iCov1,iCov2,log(k2/k1)) == 1
        
        % use the descriminant function template for C3 and C1, to 
        % evaluate the boundary u13
        if dscrmnt(x,U3,U1,iCov3,iCov1,log(k1/k3)) == 1
            
            % have found an observation which falls into C3
            % add an entry to the appropriate class array
            C3 = [C3; x];
            
        else
            
            % x indeed belongs to C1
            C1 = [C1; x];
        
        end
    
    % descriminant function template for C2 and C3
    elseif dscrmnt(x,U2,U3,iCov2,iCov3,log(k3/k2)) == 1
        
        % x indeed belongs to C1
        C2 = [C2; x];
        
    else
        
        % x belongs to C3
        C3 = [C3; x];

    end
    
end

% print the axis labels
xlabel('x_1');
ylabel('x_2');

% plot the observations in Xx, the color indicates the class to which an 
% observation was assigned (same color code as in 2.1)
plot(C1(:,1), C1(:,2), 'r.');
plot(C2(:,1), C2(:,2), 'g.');
plot(C3(:,1), C3(:,2), 'b.');

% let's plot the class boundaries
syms x1 x2
u12=-0.5*(([x1 x2] - U1)*iCov1*([x1; x2]-U1') - ([x1 x2] - U2)*iCov2*([x1; x2]-U2')) - log(k2/k1);
u13=-0.5*(([x1 x2] - U1)*iCov1*([x1; x2]-U1') - ([x1 x2] - U3)*iCov3*([x1; x2]-U3')) - log(k3/k1);
u23=-0.5*(([x1 x2] - U2)*iCov2*([x1; x2]-U2') - ([x1 x2] - U3)*iCov3*([x1; x2]-U3')) - log(k3/k2);

pu12 = ezplot(u12,[2,8,-2,6]);
h = get(gca,'children');
set(h(1),'linestyle','-.','color','b','linewidth',1);

pu13 = ezplot(u13,[2,5,4,12]);
h = get(gca,'children');
set(h(1),'linestyle','-.','color','g','linewidth',1);

pu23 = ezplot(u23,[-3,2,-2,6]);
h = get(gca,'children');
set(h(1),'linestyle','-.','color','r','linewidth',1);

% color code for the observation-class assignment
legend('C_1','C_2','C_3','u12','u13','u23');
