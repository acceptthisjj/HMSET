function [new_data,new_label]=KN_test_creat(data,label,testdata,testlabel,k,weight1,weight2)
%data数据 行样本 列特征 mxn
%label数据标签 mx1
%class数据类别个数
%k近邻个数 实际上是算上自己 近邻k-1
%两类标签 1 0

%w属性的重要程度
[m,n]=size(testdata);
Di_index=find(label==1);%找到类别索引
Dni_index=find(label==0);
Di=data(Di_index,:);%按类别分出数据集
Dni=data(Dni_index,:);%

new_data=[];
new_label=[];
for i=1:m
dest=testdata(i,:);%选择一个样本
if testlabel(i,1)==1
    nr=pdist2(dest,Di);%计算该样本和同类样本的欧式距离 1xn hxn  结果 1xh
    [num,idx]=sort(nr);%排序
    per=Di(idx(1:k),:);
    per(1,:)=per(1,:).*weight1;
    per(2:end,:)=per(2:end,:).*(weight2/(k-1));
    new_data=[new_data;per];
    new_label=[new_label;ones(k,1)];
else
    nr=pdist2(dest,Dni);
    [num,idx]=sort(nr);%排序
    per=Dni(idx(1:k),:);
    per(1,:)=per(1,:).*weight1;
    per(2:end,:)=per(2:end,:).*(weight2/(k-1));
    new_data=[new_data;per];
    new_label=[new_label;zeros(k,1)];
end
end
end