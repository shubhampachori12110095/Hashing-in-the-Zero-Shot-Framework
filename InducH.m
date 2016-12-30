function [EmbeddingS, W_RS_nmlz, sigma] = InducH(S, R, options)
% function for IMH-LEH in the paper

%InducH(anchor, trainingfeatures, options);

r = options.maxbits;
% S = anchor;
% R = trainingfeatures;

%% constuct graph on  S
[~,W_SS] = get_Z(S, S, options.k, options.sigma); 
if isfield(options,'bTrueKNN') && options.bTrueKNN  
else
    W_SS = max(W_SS,W_SS');
end

D_SS = diag(full(sum(W_SS,2))); 
M = D_SS - W_SS;

%% constuct graph on  SR
[W_RS_nmlz, W_RS,sigma] = get_Ztrain(R, S, options.s, options.sigma, options.sigma2); % This gives the probability to which class an image belongs to in the 
%training classes.

D_SR = diag(sum(W_RS,1));
T = D_SR - W_RS_nmlz'*W_RS;

%% compute eigensystem
[V, eigenvalue] = eig(M + 2*T);
[eigenvalue,order] = sort(diag(eigenvalue));
V = V(:,order);
clear order;
ind = eigenvalue> 1e-3;
if (r ~= 40)
 V = V(:,ind);
end
EmbeddingS = V(:,1:r);


end
