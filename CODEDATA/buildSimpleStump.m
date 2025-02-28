function [Dim,Dir,T,best_label,minError] = buildSimpleStump(data,label,D)
%Dim ���ְ��յ�����ά��
%Dir 
%T ���ʵ���ֵ
%best_label ������ǩ
%minError ���

% ����һ������
numSteps = 50;
% m������,ÿ��nά
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
        %-----С����ֵ��ȡֵΪ-1��--------------------
        label_temp = ones(m,1);
        label_temp(index) = -1;
        index1 = find(label_temp == label);
        errArr = ones(m,1);
        errArr(index1) = 0;
        %С����ֵ�����
        weightError = D'*errArr;
        if weightError < minError
            bestLabel = label_temp;
            minError = weightError;
            %С����ֵ�ĵ�ȡ-1��ǩ
            direction = -1;
            Dim = i; %��¼���ڵ�ά��
            thresh = threshVal;
        end      
        %-----------С����ֵ��ȡֵΪ+1��---------
        label_temp = -1*ones(m,1);
        label_temp(index) = 1;
        index1 = find(label_temp == label);
        errArr = ones(m,1);
        errArr(index1) = 0;
        %������ֵ�����
        weightError = D'*errArr;
        if weightError < minError
            bestLabel = label_temp;
            minError = weightError;
            %С����ֵ�ĵ�ȡ+1��ǩ
            direction = 1;
            Dim = i; %��¼���ڵ�ά��
            thresh = threshVal;
        end  
    end
end
Dir = direction;
T = thresh;
best_label = bestLabel;