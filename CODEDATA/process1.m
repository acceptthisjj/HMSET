function [UX,UY] = process1(X,Y,type,a,b) 
%处理数据集 把每个病人的6个片段 降维成六个片段 均值、中位数、标准差、平均绝对偏差、四分位范围、截尾均值
%a 人数 b 一个人的样本数

switch type
    case 1
        method.mode='141';
    case 2
        method.mode='251';
    case 3
        method.mode='361';
    case 4
        method.mode='all1';
end

[m, n] = size(X);
dataperone=[];
for i=1:a
    dataper=X(1+b*(i-1):b+b*(i-1),:);%选择一个病人的片段
    dataperone=cat(1,dataperone,mlconverttosix(dataper,method,b));%进行降维
end
UX=dataperone;

[m, n] = size(Y);
dataperone1=[];
switch method.mode
    case'141'
        s=2;
    case'251'
        s=2;
    case'361'
        s=2;
    case'all1'
        s=6;
end

for i=1:a
    tempY=[];
    for count=1:s
        tempY=[tempY;Y(1+b*(i-1),:)];
    end
    dataper1=tempY;%选人
    dataperone1=cat(1,dataperone1,dataper1);%匹配ddrx的标签
end
UY=dataperone1;
end