% Formation of cosine similarity for anchors for test classes. That is new
% classes. This is arranged according to the testClassesIndices.

load('testClassesIndices.mat')
for i = 1:length(testClassesIndices)
    AA(i,:) = attrClasses(testClassesIndices(i),:);
end    

anchortrainsequence = CC;

for i = 1:length(anchortrainsequence)
    BB(i,:) = attrClasses(anchortrainsequence(i),:);
end 

for i = 1:size(AA,1)
    for j = 1:size(BB,1)
        
        DD(i,j) = dot(AA(i,:),BB(j,:))  /((norm(AA(i,:))*norm(BB(j,:))));
        
    end
end    
  

cossimfortestclassesarrangedaccordingtoorder = DD;
    
m = size(DD,2) + 1;
DD(:,m) = testClassesIndices;