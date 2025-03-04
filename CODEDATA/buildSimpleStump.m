function [Dim,Dir,T,best_label,minError] = buildSimpleStump(data,label,D)
%Dim 划分按照的所属维度
%Dir 
%T 合适的阈值
%best_label 分类后标签
%minError 误差

% 设置一个步长
numSteps = 50;
% m个样本,每个n维
[m,n] = size(data);
thresh = 0;
minError = inf;
for i = 1:n
    min_dataI = min(data(:,i));
    max_dataI = max(data(:,i));
    step_add = (max_dataI - min_dataI)/numSteps;
    for j = 1:numSteps
        threshVal = min_dataI + j*step_add;
        index = find(data(:,i) <= threshVal);
        %-----小于阈值的取值为-1类--------------------
        label_temp = ones(m,1);
        label_temp(index) = -1;
        index1 = find(label_temp == label);
        errArr = ones(m,1);
        errArr(index1) = 0;
        %小于阈值的误差
        weightError = D'*errArr;
        if weightError < minError
            bestLabel = label_temp;
            minError = weightError;
            %小于阈值的点取-1标签
            direction = -1;
            Dim = i; %记录属于的维度
            thresh = threshVal;
        end      
        %-----------小于阈值的取值为+1类---------
        label_temp = -1*ones(m,1);
        label_temp(index) = 1;
        index1 = find(label_temp == label);
        errArr = ones(m,1);
        errArr(index1) = 0;
        %大于阈值的误差
        weightError = D'*errArr;
        if weightError < minError
            bestLabel = label_temp;
            minError = weightError;
            %小于阈值的点取+1标签
            direction = 1;
            Dim = i; %记录属于的维度
            thresh = threshVal;
        end  
    end
end
Dir = direction;
T = thresh;
best_label = bestLabel;