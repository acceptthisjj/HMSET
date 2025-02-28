function [dim,direction,thresh,alpha] = adaBoostTrainDs(data,label,iter)
[m,~] = size(data);
% ��ʼ��ȨֵD
D = ones(m,1)/m;
alpha = zeros(iter,1);
% ��¼T����
direction = zeros(iter,1);
% ��¼T������һ��ά��
dim = zeros(iter,1);
% ��ʼ����ֵT
thresh = zeros(iter,1);
for i = 1:iter
    [dim(i),direction(i),thresh(i),best_label,error] = ...
        buildSimpleStump(data,label,D);
    %����alpha
    alpha(i) = 0.5*log((1-error)/max(error,1e-15));
    %����ȨֵD
    D = D.*(exp(-1*alpha(i)*(label.*best_label)));
    D = D/sum(D);
end