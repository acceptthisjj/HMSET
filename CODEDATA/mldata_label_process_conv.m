%���������ݼ� �����ع����
function [datanew,labelnew] = mldata_label_process_conv(train_data,train_label,sub,sam)
%���ݴ���
datanew=[];ans1=[];
labelnew=[];
for i=1:sub
x=train_data(1+sam*(i-1):sam+sam*(i-1),:);%һ�δ�������������
y=train_label(1+sam*(i-1):sam+sam*(i-1),:);

[d1,l1]=mlperctsix(x,y);%�ع�����
r1=x;%��������
for num=1:size(d1,1)%��ÿ���ع�����������
    ans1=[ans1;sum(d1(num,:).*r1,1)]; %�˻���� ����
end
dataper=ans1;%daigai
labelper=l1;
ans1=[];%���

datanew=cat(1,datanew,dataper);
labelnew=cat(1,labelnew,labelper);
end
end