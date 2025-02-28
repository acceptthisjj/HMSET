%输入subject的sample
%返回分好2类后的一次变换指标2*6
function [U1,P1,U2,P2] = perctsix(pdata,plabel,c1,c2)
cdata1=pdata(c1,:);
cdata2=pdata(c2,:);

h1=size(cdata1,1);
h2=size(cdata2,1);
%第一次降维处理
U1(1,:)=sum(cdata1,1);%求列均值
U1(2,:)=median(cdata1,1);%中位数
U1(4,:)=std(cdata1,1);% 计算标准差，权值为1，维度为1，也就是计算列标准差
U1(5,:)=mad(cdata1);%平均绝对偏差
U1(6,:)=quantile(cdata1,0.75,1)-quantile(cdata1,0.25,1);%四分位范围
[val1,ind1]=sort(cdata1);
U1(3,:)=mean(val1(ceil(h1*0.2):ceil(h1*0.8),:),1);% 20%截尾均值

U2(1,:)=sum(cdata2,1);%求列均值
U2(2,:)=median(cdata2,1);%中位数
U2(4,:)=std(cdata2,1);% 计算标准差，权值为1，维度为1，也就是计算列标准差
U2(5,:)=mad(cdata2);%平均绝对偏差
U2(6,:)=quantile(cdata2,0.75,1)-quantile(cdata2,0.25,1);%四分位范围
[val2,ind2]=sort(cdata2);
U2(3,:)=mean(val2(ceil(h2*0.2):ceil(h2*0.8),:),1);% 20%截尾均值

tempY=[];
for i=1:6
        tempY=[tempY;plabel(1,:)];
end
P1=tempY;
P2=tempY;
end