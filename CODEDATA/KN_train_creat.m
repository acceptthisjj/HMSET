function [new_data,new_label]=KN_train_creat(data,label,k,weight1,weight2)
%data���� ������ ������ mxn
%label���ݱ�ǩ mx1
%class����������
%k���ڸ��� ʵ�����������Լ� ����k-1
%�����ǩ 1 0

%w���Ե���Ҫ�̶�
[m,n]=size(data);
Di_index=find(label==1);%�ҵ��������
Dni_index=find(label==0);
Di=data(Di_index,:);%�����ֳ����ݼ�
Dni=data(Dni_index,:);%

new_data=[];
new_label=[];
for i=1:m
dest=data(i,:);%ѡ��һ������
if label(i,1)==1
    nr=pdist2(dest,Di);%�����������ͬ��������ŷʽ���� 1xn hxn  ��� 1xh
    [num,idx]=sort(nr);%����
    %��Ȩ��
    per=Di(idx(1:k+1),:);
    per(1,:)=per(1,:).*weight1;
    per(2:end,:)=per(2:end,:).*(weight2/(k));
    new_data=[new_data;per];
    new_label=[new_label;ones(k+1,1)];
else
    nr=pdist2(dest,Dni);
    [num,idx]=sort(nr);%����
    per=Dni(idx(1:k+1),:);
    per(1,:)=per(1,:).*weight1;
    per(2:end,:)=per(2:end,:).*(weight2/(k));
    new_data=[new_data;per];
    new_label=[new_label;zeros(k+1,1)];
end
end
end