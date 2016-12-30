% Got by modifying Wei Liu's codes
function [Z_nmlz, Z, sigma] = get_Z(X, Anchor, s, sigma)

% get_Z(traindata, anchor, options.s, options.sigma);


% This function gives Z_nmlz and Z which are the normailized and non
% normalized probabilites to which the data points may belong to the s
% anchors. That is they give the weights to which the new datapoints points
% may belong to which anchor. Thus it helps to calcaulte the weighted
% embedding.




[n,~] = size(X);
m = size(Anchor,1);

%% get Eucilidian distance

% This will calculate the distance  between our datapoints and anchor
% points in our case.

% Distance matrix 

% Dis matrix gives the distance of each point in the datafile X form the anchors.

if n <= 1e5
    Dis = EuDist2(X,Anchor,0);
else
    Dis = zeros(n, m);
    l = floor(n / 1e5); r = mod(n, 1e5);
    for i = 1 : l
        Xi = X((i-1)*1e5 + [1:1e5], :);
        Dis((i-1)*1e5 + [1:1e5], :) = EuDist2(Xi,Anchor,0);  
%         pause(1);
    end
    clear Xi;
    if r > 0
        Dis(l*1e5 + 1 : end,:) = EuDist2(X(l*1e5 +1 : end,:),Anchor,0);
    end
end
clear X;
clear Anchor;
% display('0.....');

%% get Z

% display('1.....');
val = zeros(n,s);
pos = val;

% We are collecting top five anchors and their distances of a new data
% points. These s (here in the original code it is 5) anchors are selected according to which the distance
% between a new data point is lowerst. This distance gives helps to find us
% the manifold embedding of a new point with respect to the anchors.
% Whether the embedding is of training point or the testing point. 

for i = 1:s
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]'; % This tep changes the lowest distance to the highest so that second minimum point could be detected. This follows till the top s anchors are detected.
    Dis(tep) = 1e60; 
end
clear Dis;
clear tep;



if sigma == 0
   sigma = mean(val(:,s).^0.5); % Only for the sth value we have calculate mean because mean(v(:,s)) is the largest among
   % mean(val(:,1)),mean(val(:,2)), .... , mean(val(:,s-1)) etc. Hence
   % standard deviation is calucalted with respect to it only.
%     sigma = mean(mean(val)).^0.5;
end
val = exp(-val/(1/1*sigma^2)); % % This converts the distance into the probability or the weight measure.

% This value is then used to calculate the new maniflod embedding of the
% new data.


%Below Z contains the weight values of the new data points with respect to
%the anchor points. It will be zero else only where the weights with
%trepsect to which new data point has minimum distance is kept non - zero
%and the rest all are zero. To save the space and construct the matrix
%efficiently, Z matrix is converted into sparse form.
if nargout >= 2
    Z = zeros(n,m);
    tep = (pos-1)*n+repmat([1:n]',1,s);
    Z([tep]) = [val];
    Z = sparse(Z);
end
%% get normalized Z
val_nmlz = repmat(sum(val,2).^-1,1,s).*val;
% This is normalized probability values. That is the probability values are
% normalized so that sum of the probabilites or weights of a data point
% from the s anchors is equal to 1.

% normalize %val_nmlz = bsxfun(@rdivide, val, sum(val,2)); %

%Below Z_nmlz contains the weight values of the new data points with respect to
%the anchor points. It will be zero else only where the weights with
%trepsect to which new data point has minimum distance is kept non - zero
%and the rest all are zero. To save the space and construct the matrix
%efficiently, Z_nmlz matrix is converted into sparse form.

clear val;
Z_nmlz = zeros(n,m);
if ~exist('tep', 'var')
    tep = (pos-1)*n+repmat([1:n]',1,s);
end
Z_nmlz([tep]) = [val_nmlz];
Z_nmlz = sparse(Z_nmlz);
clear val_nmlz;
clear tep;
clear pos;
% display('3.....');

end

