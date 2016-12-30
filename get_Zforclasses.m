% Got by modifying Wei Liu's codes
function [Z_nmlz, Z] = get_Zforclasses(cossimfortestclasses, anchor, s)

% here similaritymatrix is a cosine similarity matrix. anchors are the
% clusters. s is how many number of anchors you want to take for the
% closest assignment so that you can find the new embedding using weighted
% anchors. Maximum value of s is the number of seen classes, minimum is 1.

%
% This function gives Z_nmlz and Z which are the normailized and non
% normalized probabilites to which the data points may belong to the s
% anchors. That is they give the weights to which the new datapoints points
% may belong to which anchor. Thus it helps to calcaulte the weighted
% embedding.

[n,~] = size(cossimfortestclasses);
m = size(anchor,1);


% get Z

val = zeros(n,s);
pos = val;

% We are collecting top five anchors and their distances of a new data
% points. These s (here in the original code it is 5) anchors are selected according to which the distance
% between a new data point is lowerst. This distance gives helps to find us
% the manifold embedding of a new point with respect to the anchors.
% Whether the embedding is of training point or the testing point. 

for i = 1:s
    [val(:,i),pos(:,i)] = max(cossimfortestclasses,[],2);
    tep = (pos(:,i)-1)*n+[1:n]'; % This tep changes the highest probability to the lowestt so that second maximum point could be detected. This follows till the top s anchors are detected.
    cossimfortestclasses(tep) = -1e60; 
end

% This value is then used to calculate the new maniflod embedding of the
% new data.


%Below Z contains the weight values of the new data classes with respect to
%the anchor points. It will be zero else only where the weights with
%repsect to which new data point has minimum distance is kept non - zero
%and the rest all are zero. To save the space and construct the matrix
%efficiently, Z matrix is converted into sparse form.

if nargout >= 2
    Z = zeros(n,m);
    tep = (pos-1)*n+repmat([1:n]',1,s);
    Z([tep]) = [val];
    Z = sparse(Z);
end
% get normalized Z
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
end
