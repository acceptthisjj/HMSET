function [new_data,new_label]=KN_cedi_train_creat(data,label,k,weight1,weight2)
%data数据 行样本 列特征 mxn
%label数据标签 mx1
%class数据类别个数
%k近邻个数（每个包络k+1个样本）
%两类标签 1 0
%weight1 主样本权重
%weight2 k*副样本权重

%w属性的重要程度
[m,n]=size(data);
Di_index=find(label==1);%找到类别索引
Dni_index=find(label==0);
Di=data(Di_index,:);%按类别分出数据集
Dni=data(Dni_index,:);%

new_data=[];
new_label=[];

%计算同类数据的测地距离图
mapDi=KN_chedi(Di,2);
mapDni=KN_chedi(Dni,2);
%排序
[mapnumDi,mapidxDi]=sort(mapDi,2);
[mapnumDni,mapidxDni]=sort(mapDni,2);
for i=1:m
    %定位该样本
    if label(i,1)==1
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