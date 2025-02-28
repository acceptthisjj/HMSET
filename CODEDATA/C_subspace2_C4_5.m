function [labelval,labeltest]=C_subspace2_C4_5(numtree,trainX,trainY,validX,testX,envelope)
%获得多个重采样子空间训练弱分类器并获得测试标签
%随机抽样80%样子空间
%各子空间预测结果
%分别对验证集和测试集
labelval=[];
labeltest=[];
[b1,m1,n1]=unique(trainX,'rows','stable'); %不自动排序  提出后数据 选择的样本索引
%剔除重复
dataX=b1;
dataY=trainY(m1,:);

m=size(dataY,1);
for i=1:numtree
    A=randperm(m,round(m*0.8));
    subtrainx= dataX(A,:);
    subtrainY= dataY(A,:);
    prey=C4_5(subtrainx',subtrainY',validX',5);
    prey=prey';%一行变一列
    va_prey=B_compute_label(prey,envelope);%将包络预测标签转成原样本预测标签
    labelval=[labelval,va_prey];
    
    prey=C4_5(subtrainx',subtrainY',testX',5);
    prey=prey';%一行变一列
    te_prey=B_compute_label(prey,envelope);%将包络预测标签转成原样本预测标签
    labeltest=[labeltest,te_prey];    
end
end