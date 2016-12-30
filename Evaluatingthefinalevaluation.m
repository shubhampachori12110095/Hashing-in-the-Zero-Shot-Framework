
clc; clear all; close all;

load('realtestClassLabels.mat')
load('realtrainLabels.mat')


reallabels = [realtrainLabels;realtestClassLabels];

p = randperm(length(reallabels));

training_final_evaluation = reallabels(p(2001:end));
testing_final_evaluation = reallabels(p(1:2000));
B = compactbit(H); 
tB = compactbit(tH);
BB = [B;tB];

evaltrainBB = BB(p(2001:end));
evaltestBB = BB(p(1:2000));

cateTrainTest = bsxfun(@eq, training_final_evaluation, testing_final_evaluation');

hammTrainTest = hammingDist(evaltestBB, evaltrainBB)';

Ret = (hammTrainTest <= hammRadius+0.00001);

[cateP, cateR] = evaluate_macro(cateTrainTest, Ret); % This gives the precision and recall between the true output
    % and the obtained output through training.
cateF1(i) = F1_measure(cateP, cateR); % Fmeasure using precision and recall.
clear cateP cateR
    % get hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest,1); % This sorts that to which training data does the new testing data belongs to.
% That si given the rank to which training point the distance of new
% testing point is lowest.
[cateMAP(i)] = cat_apcal(training_final_evaluation,testing_final_evaluation,HammingRank);



% 
