function [labelarr]=subspace2_C4_5(numtree,trainX,trainY,testX,envelope,class1,class2)
%获得多个重采样子空间训练弱分类器并获得测试标签
%随机抽样70%样子空间
%各子空间预测结果
labelarr=[];

[b1,m1,n1]=unique(trainX,'rows','stable'); %不自动排序  提出后数据 选择的样本索引
%剔除重复
dataX=b1;
dataY=trainY(m1,:);

m=size(dataY,1);
for i=1:numtree
    A=randperm(m,round(m*0.7));
    subtrainx= dataX(A,:);
    subtrainY= dataY(A,:);
%     a1=length(subtrainY==0)
%     b1=length(subtrainY==1)
    
    prey=C4_5(subtrainx',subtrainY',testX',5);
    prey=prey';%一行变一列
    re_prey=compute_label(prey,envelope,class1,class2);%将包络预测标签转成原样本预测标签
    labelarr=[labelarr,re_prey];
end
end

%重采样的方式 C4.5报错 训练集中有重复样本？太多？
% function [labelarr]=subspace_C4_5(numtree,trainX,trainY,testX,envelope,class1,class2)
% %获得多个重采样子空间训练弱分类器并获得测试标签
% %重采样子空间
% [bootstat,bootsam] = bootstrp(numtree, @(x) mean(x), trainX);%重采样
% %各子空间预测结果
% labelarr=[];
% for i=1:size(bootsam,2)
%     subtrainx= trainX(bootsam(:,i),:);
%     subtrainY= trainY(bootsam(:,i),:);
%     prey=C4_5(subtrainx',subtrainY',testX',5);
%     prey=prey';%一行变一列
%     re_prey=compute_label(prey,envelope,class1,class2);
%     labelarr=[labelarr,re_prey];
% end
% end
