function [new_data,new_label]=B_KN_cedi_train_creat(data,label,k,weight1,weight2)
%data数据 行样本 列特征 mxn
%label数据标签 mx1
%class数据类别个数
%k近邻个数（每个包络k+1个样本）
%多类标签
%weight1 主样本权重
%weight2 k*副样本权重
%w属性的重要程度
new_data=[];
new_label=[];
[m,n]=size(data);
uni=unique(label);%从小到大 排序
for i=1:size(uni,1)
    A_index{i}=find(label==uni(i));
    A_data{i}=data(A_index{i},:);
    map=KN_chedi(A_data{i},2);
   [mapnum{i},mapidx{i}]=sort(map,2);%排序
end
for i=1:m
    fkey=find(uni==label(i,1));
        idx_inDi=find(A_index{fkey}==i);%获取在数据位置
        %找到该样本的k近邻测地权重样本
        per=A_data{fkey}(mapidx{fkey}(idx_inDi,1:k+1),:);
        per(1,:)=per(1,:).*weight1;
        per(2:end,:)=per(2:end,:).*(weight2/(k));
        new_data=[new_data;per];
        new_label=[new_label;repelem(uni(fkey,1),k+1,1)];
end
end