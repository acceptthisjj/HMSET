function [new_data,new_label]=KN_cedi_test_creat(data,label,testdata,testlabel,k,weight1,weight2)
%data数据 行样本 列特征 mxn
%label数据标签 mx1
%class数据类别个数
%k近邻个数 实际上是算上自己 近邻k-1
%两类标签 1 0
kn_nearcount=2;
%w属性的重要程度
[m,n]=size(testdata);
%统合 前m个是测试样本
all_label=[testlabel;label];
all_data=[testdata;data];

Di_index=find(all_label==1);%找到类别索引
Dni_index=find(all_label==0);
Di=all_data(Di_index,:);%按类别分出数据集
Dni=all_data(Dni_index,:);%

new_data=[];
new_label=[];
%计算同类数据的测地距离图
mapDi=KN_chedi(Di,kn_nearcount);
mapDni=KN_chedi(Dni,kn_nearcount);
%排序
[mapnumDi,mapidxDi]=sort(mapDi,2);
[mapnumDni,mapidxDni]=sort(mapDni,2);

for i=1:m%找前m个样本的加权近邻包络样本
    %定位该样本
    if all_label(i,1)==1
        idx_inDi=find(Di_index==i);%获取在该类数据集中的位置
        %找到该样本的k近邻测地权重样本
        per=Di(mapidxDi(idx_inDi,1:k+1),:);
        per(1,:)=per(1,:).*weight1;
        per(2:end,:)=per(2:end,:).*(weight2/(k));
        new_data=[new_data;per];
        new_label=[new_label;ones(k+1,1)];
    else
        idx_inDni=find(Dni_index==i);%获取在该类数据集中的位置
        %找到该样本的k近邻测地权重样本
        per=Dni(mapidxDni(idx_inDni,1:k+1),:);
        per(1,:)=per(1,:).*weight1;
        per(2:end,:)=per(2:end,:).*(weight2/(k));
        new_data=[new_data;per];
        new_label=[new_label;zeros(k+1,1)];
    end
end
end