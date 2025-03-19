% 假设 trainData 是训练集特征矩阵，trainLabels 是训练集标签向量
% testData 是测试集特征矩阵，每一行是一个测试样本
function [new_data,new_label]=B_KN_cedi_test_creatN1(trainData,trainLabels,testData,testlabel,k,weight1,weight2)
% 计算测试集中每个样本与训练集中所有样本的欧氏距离
numTestSamples = size(testData, 1);
numNearestNeighbors = k;
 new_data=[];
 new_label=[];
 t=[];
% 预先分配空间来存储结果
nearestNeighborsIdx = zeros(numTestSamples, numNearestNeighbors);
nearestNeighborsLabels = zeros(numTestSamples, numNearestNeighbors);
 
for i = 1:numTestSamples
    % 计算测试样本与所有训练样本的欧氏距离
   % distances = sqrt(sum((trainData - testData(i, :)).^2, 2));
    distances=KN_chediT(testData,trainData,k);
    % 获取最近的 numNearestNeighbors 个训练样本的索引
    [~, sortedIdx] = sort(distances(i,:));
    nearestIdx = sortedIdx(1:numNearestNeighbors);
    
    nearestLabels = trainLabels(nearestIdx);

    %labelCounts = histcounts(nearestLabels, [min(trainLabels)-0.5, max(trainLabels)+0.5, 1]);
    [uniqueValues, ~, idx] = unique(nearestLabels);
    counts = accumarray(idx, 1);
    [maxValue, maxLabelIdx] = max(counts);
     mostFrequentNumber = uniqueValues(maxLabelIdx);
     maxLabel =mostFrequentNumber;
    
    nearestNeighborsT = nearestIdx(nearestLabels ==   mostFrequentNumber);
    if numel(nearestNeighborsT) < numNearestNeighbors
        % 找出所有标签为 maxLabel 的训练样本索引
        maxLabelIdxTrain = find(trainLabels ==   maxLabel);
        
        % 计算测试样本与这些标签为 maxLabel 的训练样本的距离
%         distancesToMaxLabel = sqrt(sum((trainData(maxLabelIdxTrain, :) - testData(i, :)).^2, 2));
         distancesToMaxLabel=KN_chediT(testData,trainData(maxLabelIdxTrain, :),2);
        % 获取最近的 numNearestNeighbors - numel(nearestNeighborsIdx(i, :)) 个样本的索引
        [~, additionalSortedIdx] = sort(distancesToMaxLabel(i,:));
         nearestIdx = additionalSortedIdx(1:numNearestNeighbors);

   
      nearestNeighborsIdx(i, :)=nearestIdx;
      %nearestNeighborsLabels(i, :)=nearestLabels(nearestLabels ==  mostFrequentNumber);
       per(1,:)=testData(i,:).*weight1;       
        per(2:k+1,:)=trainData(nearestNeighborsIdx(i,:),:).*weight2;
        M=[repelem(maxLabel,k+1,1)];
    else
    nearestNeighborsIdx(i, :) = nearestIdx(nearestLabels ==   mostFrequentNumber);
    nearestNeighborsLabels(i, :) = nearestLabels(nearestLabels ==  mostFrequentNumber);
     per(1,:)=testData(i,:).*weight1;       
        per(2:k+1,:)=trainData(nearestNeighborsIdx(i,:),:).*weight2;
        M=[nearestNeighborsLabels(i);(nearestNeighborsLabels(i,:))'];
     end

        new_data=[new_data;per];
        
        new_label=[new_label;M];
        t=[t;repelem(testlabel(i),k+1,1)];
end
end
