function [new_data,new_label]=B_KN_cedi_train_creat(data,label,k,weight1,weight2)
%data���� ������ ������ mxn
%label���ݱ�ǩ mx1
%class����������
%k���ڸ�����ÿ������k+1��������
%�����ǩ
%weight1 ������Ȩ��
%weight2 k*������Ȩ��
%w���Ե���Ҫ�̶�
new_data=[];
new_label=[];
[m,n]=size(data);
uni=unique(label);%��С���� ����
for i=1:size(uni,1)
    A_index{i}=find(label==uni(i));
    A_data{i}=data(A_index{i},:);
    map=KN_chedi(A_data{i},2);
   [mapnum{i},mapidx{i}]=sort(map,2);%����
end
for i=1:m
    fkey=find(uni==label(i,1));
        idx_inDi=find(A_index{fkey}==i);%��ȡ������λ��
        %�ҵ���������k���ڲ��Ȩ������
        per=A_data{fkey}(mapidx{fkey}(idx_inDi,1:k+1),:);
        per(1,:)=per(1,:).*weight1;
        per(2:end,:)=per(2:end,:).*(weight2/(k));
        new_data=[new_data;per];
        new_label=[new_label;repelem(uni(fkey,1),k+1,1)];
end
end