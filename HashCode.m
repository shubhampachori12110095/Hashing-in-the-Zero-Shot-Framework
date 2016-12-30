clear all ; close all; clc;

% These are the parameters. No logic is in it.

addpath ./tSNE
%
%settings
%codeLen = [16, 32, 64]; %number of hash bits.

codeLen = [16,24,32,40];
hammRadius = 1;
m = length(codeLen);
% isgnd =1;
load anchorarrangedaccordingtoclassnumbers.mat % Have to find the new anchors as mentioned in the paper.

%"Semi-supervised Zero-Shot Learning by a Clustering-based Approach"
% anchor_numbers = 40;

load binAttrClasses.mat % Remember the binAttrClasses are arranged according to teh real class number.
load trainlabels.mat
load trainingfeatures.mat
load testlabels.mat
load testingfeatures.mat
load datasetfeatures_AwA.mat
load trainInstancesLabels.mat
load testInstancesLabels.mat
load cosinesimilaritybwclasses.mat
load testClassesIndices.mat
load trainClassesIndices.mat
load cossimfortestclasses.mat %(arranged according to the the order of test Indices)
load anchortrainsequence.mat
load normalizedanchors.mat
load normalizedtestingfeatures.mat
load normalizedtrainingfeatures.mat
load trainlabelsaccordingtotrainanchors.mat
% normalizedtrainingfeatures = normalize(trainingfeatures); % Normalize all feature vectors in trainingdata to unit length
% normalizedtestingfeatures = normalize(testingfeatures); % Normalize all feature vectors in testingdata to unit 
% normalizedanchors = normalized(anchor);

%%
% Initialization. % This decides the parameters that we could choose. The parameters are
% different for different methods. Like the options are different for
% method = 'IMH-tSNE' and method = 'IMH-LE'. These options are manifold
% embedding.

method = 'IMH-tSNE';
%method = 'IMH-LE';
% 'IMH-LE'
display([method ': ']);
options = InitOpt(method); 

%%

for i = 1 : m
    display(['learn ' num2str(codeLen(i)) ' bits...']);
%     options.nbits = codeLen(i); % This is the length of the code bits
    options.maxbits = codeLen(i); % This is the maximum length of the code bits. This is equal to the number of bits only.
    % hashing
    switch method
        
        case 'IMH-GPLVM'
            
            options.nbits = codeLen(i); % THis is the length of the code bits
% 
            [Embeddinganchor, mapping] = compute_mapping(anchor,'GPLVM', options.nbits);
            
            % get the manigold embedding for training points.
            [Z_nmlz_train,Z_train, sigma,pos] = get_Ztrain(normalizedtrainingfeatures, anchor, options.s, options.sigma,options.sigma2);
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;          
            
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                                    
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);
               
        case 'IMH-NCA'
            
            options.nbits = codeLen(i); % THis is the length of the code bits
% 
            [Embeddinganchor, mapping] = compute_mapping(anchor,'NCA', options.nbits,0);
            
            % get the manigold embedding for training points.
            [Z_nmlz_train,Z_train, sigma,pos] = get_Ztrain(normalizedtrainingfeatures, anchor, options.s, options.sigma,options.sigma2);
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;          
            
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                                    
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);
        
        
        
        case 'IMH-KernelPCA'
            options.nbits = codeLen(i); % THis is the length of the code bits
% 
            [Embeddinganchor, mapping] = compute_mapping(anchor,'KernelPCA', options.nbits,'poly');
            
            % get the manigold embedding for training points.
            [Z_nmlz_train,Z_train, sigma,pos] = get_Ztrain(normalizedtrainingfeatures, anchor, options.s, options.sigma,options.sigma2);
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;          
            
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                                    
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);      
        
        case 'IMH-Isomap'
            options.nbits = codeLen(i); % THis is the length of the code bits
% 
            [Embeddinganchor, mapping] = compute_mapping(anchor,'Isomap', options.nbits,[]);
            
            % get the manigold embedding for training points.
            [Z_nmlz_train,Z_train, sigma,pos] = get_Ztrain(normalizedtrainingfeatures, anchor, options.s, options.sigma,options.sigma2);
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;          
            
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                                    
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);
               
        case 'IMH-LLE'
            options.nbits = codeLen(i); % THis is the length of the code bits
% 
            [Embeddinganchor, mapping] = compute_mapping(anchor,'LLE', options.nbits,[]);
            
            % get the manigold embedding for training points.
            [Z_nmlz_train,Z_train, sigma,pos] = get_Ztrain(normalizedtrainingfeatures, anchor, options.s, options.sigma,options.sigma2);
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;          
            
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                                    
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);
        
        case 'IMH-LE'
            options.nbits = codeLen(i); % This is the length of the code bits
            
            % get embedding for anchor points
            [Embeddinganchor,Z_nmlz_train,sigma] = InducH(anchor, trainingfeatures, options);
            
            % get the manigold embedding for training points.
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;            
                    
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                  
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);
            
            
            clear Embedding EmbeddingX tEmbedding Z_RS tZ;
        case 'IMH-tSNE'
            % get embedding for anchor points
            options.nbits = codeLen(i); % THis is the length of the code bits
            [Embeddinganchor] = tSNEH(anchor, options); % This produces the tSNE Embedding of the size of the bit length of the anchors. That is k means clusters centers.         
            
            % get the manigold embedding for training points.
            [Z_nmlz_train,Z_train, sigma,pos] = get_Ztrain(normalizedtrainingfeatures, anchor, options.s, options.sigma,options.sigma2);
            Embeddingtrain = Z_nmlz_train*Embeddinganchor;
            H = Embeddingtrain > 0;          
            
            % Finding the embedding of test classes. Arrange them in the
            % order.          
            [Z_nmlz_classes, Zclasses] = get_Zforclasses(cossimfortestclasses, anchor,options.s);
%           [Znmlz,Z, sigma] = get_Zforclasses(traindata, anchor, options.s, options.sigma);
            Embeddingtestclasses = Z_nmlz_classes*Embeddinganchor;
            Embeddinganchor(:,(size(Embeddinganchor,2)+1)) = anchortrainsequence;
            Embeddingtestclasses(:,(size(Embeddingtestclasses,2)+1)) = testClassesIndices;
            newEmbedding = [Embeddinganchor;Embeddingtestclasses];  
            qqq = newEmbedding(:,size(newEmbedding,2));
            [sss,lll] = sort(qqq);
            
            % This will be our embedding for all classes arranged in a
            % proper sequence according to their real classes numbers of
            % classes.
            
            Embedding = zeros(size(newEmbedding,1),(size(newEmbedding,2)-1));
            
            for ll = 1:size(lll,1)
               Embedding(ll,:) = newEmbedding(lll(ll),1:(end-1));
            end    
                                    
            % This  get_Z function gives Z_nmlz and Z which are the normailized and non
            % normalized probabilites to which the data points may belong to the s
            % anchors. That is they give the weights to which the new datapoints points
            % may belong to which anchor. Thus it helps to calcaulte the weighted
            % embedding
            
            % From here main editing begins find the probability to which
            % class the testing instance instance belongs to. And then find
            % the embedding using these anchors.
            
            [tZ_nmlz,tZ] = get_Zfortest(options.s,options.sigma2); % The same sigma obtained in the step above for  is used.
            tEmbedding = tZ_nmlz*Embeddingtestclasses(:,(1:end-1));
            tH = tEmbedding > 0; % Will give the logical vector where the testing embedding is greater than zero. This will be our binary hahs.
            
            hashtrainclasses = (Embeddinganchor(:,(1:end-1))>0);
            hashtestclasses = (Embeddingtestclasses(:,(1:end-1))>0);
            
            clear Embedding EmbeddingX tEmbedding Z tZ;
            
    end
    % evaluation
    display('Evaluation...');
    
    %%
    B = compactbit(H); % Get the compact representation of the codes. The compact representation will be (size_of_hashcode/8); Thus because of uin8 we will get the unique code.
    tB = compactbit(tH); %This is just the compact representation of our hash codes of testing dataset. his is obtained using uint8 values thus compressing 8 bits into 1 unit8.
    trainclassesB = compactbit(hashtrainclasses);
    testclassesB = compactbit(hashtestclasses);
    
    hammTrain = hammingDist(trainclassesB,B)';
    hammTest = hammingDist( testclassesB,tB)'; % Calculating the hamming distance between the testing and training dataset.
    %That is each point of the training data's distance is calculated with respect to the testing data's and a matrix 
    % is formed giving hamming distance
    
    
    for kk = 1:size(hammTrain,1)
        aaaa{kk} = find(hammTrain(kk,:) == min(hammTrain(kk,:)));
    end
    logicalarraytrain = zeros(size(hammTrain,1),1);
    for kk = 1:size(logicalarraytrain)
        logicalarraytrain(kk) = ismember(trainlabelsaccordingtotrainanchors(kk),aaaa{kk});
    end
    
    MAPtrain= mean(logicalarraytrain);
    
    for kk = 1:size(hammTest,1)
        bbbb{kk} = find(hammTest(kk,:) == min(hammTest(kk,:)));
    end
    
    logicalarraytest = zeros(size(hammTest,1),1);
    for kk = 1:size(logicalarraytest)
        logicalarraytest(kk) = ismember(testInstancesLabels(kk),bbbb{kk});
    end
    MAPtest = mean(logicalarraytest);
    
    output(i,1) = MAPtrain;
    output(i,2) = MAPtest;
    
    display(' Final Evaluation...');
    %%
    load('realtestClassLabels.mat')
    load('realtrainLabels.mat')
    load('reallabels.mat')
    load('permutationofreallabels.mat')
    %%
    training_final_evaluation = reallabels(permutationofreallabels(2001:length(permutationofreallabels)));
    testing_final_evaluation = reallabels(permutationofreallabels(1:2000));
    
    BB = [B;tB];
    evaltrainBB = BB(permutationofreallabels(2001:end),:);
    evaltestBB = BB(permutationofreallabels(1:2000),:);
    
    cateTrainTest = bsxfun(@eq, training_final_evaluation, testing_final_evaluation');
    
    hammTrainTest = hammingDist(evaltestBB, evaltrainBB)';

    Ret = (hammTrainTest <= hammRadius+0.00001);
    
    [cateP, cateR] = evaluate_macro(cateTrainTest, Ret); % This gives the precision and recall between the true output
        % and the obtained output through training.
    cateF1(i) = F1_measure(cateP, cateR); % Fmeasure using precision and recall.
    cateP1(i) = cateP;
    cateR1(i) = cateR;
    clear cateP cateR
        % get hamming ranking: MAP
    [~, HammingRank]=sort(hammTrainTest,1); % This sorts that to which training data does the new testing data belongs to.
    % That si given the rank to which training point the distance of new
    % testing point is lowest.
    [cateMAP(i)] = cat_apcal(training_final_evaluation,testing_final_evaluation,HammingRank);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    %%
    
%     Ret = (hammTrainTest <= hammRadius+0.00001); % We are only considering the training sets distances with test datasets whose 
%     % distance is less than certian threshold. This threshold  here is
%     % hammingRadius + 0.00001
%     % get hash lookup using hamming ball
%     % The following code evaluate_macro gives the precisona and recall for
%     % retrieved and relevant training documents. That is precisiona and
%     % recall with respect to the real ouptut that should come and came from
%     % the embedding.
%     [cateP, cateR] = evaluate_macro(cateTrainTest, Ret); % This gives the precision and recall between the true output
%     % and the obtained output through training.
%     cateF1(i) = F1_measure(cateP, cateR); % Fmeasure using precision and recall.
%     clear cateP cateR
%     % get hamming ranking: MAP
%     [~, HammingRank]=sort(hammTrainTest,1); % This sorts that to which training data does the new testing data belongs to.
%     % That si given the rank to which training point the distance of new
%     % testing point is lowest.
%     [cateMAP(i)] = cat_apcal(traingnd,testgnd,HammingRank); % This gives the MAP measure of the training and testing data.
end

% for i = 1:size(hammTrain,1)
%     aaaa{i} = find(hammTrain(i,:) == min(hammTrain(i,:)));
% end
% logicalarraytrain = zeros(size(hammTrain,1),1);
% for j = 1:size(logicalarraytrain)
%     logicalarraytrain(j) = ismember(trainInstancesLabels(j),aaaa{j});
% end
% mean(logicalarraytrain);
% 
% 
% 
% 
% 
% save(['results/',dataset,'_',method],  'cateMAP',  'cateF1');
% clear cateMAP cateF1
