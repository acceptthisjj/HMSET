%对输入数据集 进行重构卷积
function [datanew,labelnew] = mldata_label_process_conv(train_data,train_label,sub,sam)
%数据处理
datanew=[];ans1=[];
labelnew=[];
for i=1:sub
x=train_data(1+sam*(i-1):sam+sam*(i-1),:);%一次处理受试者样本
y=train_label(1+sam*(i-1):sam+sam*(i-1),:);

[d1,l1]=mlperctsix(x,y);%重构处理
r1=x;%分类数据
for num=1:size(d1,1)%对每个重构样本卷积求和
    ans1=[ans1;sum(d1(num,:).*r1,1)]; %乘机求和 反卷
end
dataper=ans1;%daigai
labelper=l1;
ans1=[];%清空

datanew=cat(1,datanew,dataper);
labelnew=cat(1,labelnew,labelper);
end
end