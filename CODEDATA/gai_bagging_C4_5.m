function [labelval,labeltest]=gai_bagging_C4_5(numtree,trainX,trainY,validX,testX,percent)
%按比例抽取训练集构造分类器
%获取各个基分类器的预测标签
labelval=[];
labeltest=[];

% [b1,m1,n1]=unique(trainX,'rows','stable'); %不自动排序  提出后数据 选择的样本索引
% %剔除重复
% dataX=b1;
% dataY=trainY(m1,:);
% m=size(dataY,1);

for i=1:numtree
%     A=randperm(m,round(m*percent));%随机抽？可能再小比例时导致分类失调
%     subtrainx= dataX(A,:);
%     subtrainY= dataY(A,:);
    [subtrainx,subtrainY]=splitbaseclass(trainX,trainY,percent);%按比例划分
    va_prey=Use_C4_5(subtrainx',subtrainY',validX',5,10);
    labelval=[labelval,va_prey'];
    te_prey=Use_C4_5(subtrainx',subtrainY',testX',5,10);
    labeltest=[labeltest,te_prey'];    
end

end