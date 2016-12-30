
clc; clear all; close all;

load('parametersoflambdaandgamma.mat')
load('binAttrClasses.mat') %  binAttributes mein jo values hai woh class ke numbers ke hisaab se hi hai. That is original class number hi serial number hai isme. Not any order according to the names.
load('alpha_weightsforourcase.mat')
load('KTest.mat')
load('Stest.mat')
load('realclasslabelsarrangedaccordingtodataset.mat')
load('testlabelswithoriginalclasses.mat')
pred = Stest*Alpha'*KTest';
pred = pred';
%%

s = 5; %no. of classes you want with which we want the test output to be closest to. That is number of classes with which 
n = size(pred,1);
val = zeros(n,s);
pos = val;

%

%%
% This gives the top s predictions for each testing instance.
for i = 1:s 
    [val(:,i),pos(:,i)] = max(pred,[],2);
    tep = (pos(:,i)-1)*n+[1:n]'; % This tep changes the highest probability to the lowestt so that second maximum point could be detected. This follows till the top s anchors are detected.
    pred(tep) = 0; 
end


%% The following code calculates the accuracy of the algorithm. Whether the correct class is in the top s classes or not.

classificationoutput = zeros(size(pred,1),1);
for i = 1:length(classificationoutput)
     classificationoutput(i) = ismember(testInstancesLabels(i),pos(i,1));
     %classificationoutput(i) = ismember(testlabelswithoriginalclasses(i),originalclasstestoutput(i,:));   
end   

% Here r is the accuracy with which the class  
r=mean(classificationoutput);