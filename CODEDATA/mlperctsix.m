%输入subject的sample
%返回重构卷积样本
function [U1,P1] = mlperctsix(pdata,plabel)
cdata1=pdata;
h1=size(cdata1,1);
%第一次降维处理
U1(1,:)=mean(cdata1,1);%求列均值
U1(2,:)=median(cdata1,1);%中位数
U1(4,:)=std(cdata1,1);% 计算标准差，权值为1，维度为1，也就是计算列标准差
U1(5,:)=mad(cdata1);%平均绝对偏差
U1(6,:)=quantile(cdata1,0.75,1)-quantile(cdata1,0.25,1);%四分位范围
[val1,ind1]=sort(cdata1);
U1(3,:)=mean(val1(ceil(h1*0.2):ceil(h1*0.8),:),1);% 20%截尾均值
P1=plabel(1:6,:);
end