function [new_data,new_label]=KN_cedi_test_creat(data,label,testdata,testlabel,k,weight1,weight2)
%data���� ������ ������ mxn
%label���ݱ�ǩ mx1
%class����������
%k���ڸ��� ʵ�����������Լ� ����k-1
%�����ǩ 1 0
kn_nearcount=2;
%w���Ե���Ҫ�̶�
[m,n]=size(testdata);
%ͳ�� ǰm���ǲ�������
all_label=[testlabel;label];
all_data=[testdata;data];

Di_index=find(all_label==1);%�ҵ��������
Dni_index=find(all_label==0);
Di=all_data(Di_index,:);%�����ֳ����ݼ�
Dni=all_data(Dni_index,:);%

new_data=[];
new_label=[];
%����ͬ�����ݵĲ�ؾ���ͼ
mapDi=KN_chedi(Di,kn_nearcount);
mapDni=KN_chedi(Dni,kn_nearcount);
%����
[mapnumDi,mapidxDi]=sort(mapDi,2);
[mapnumDni,mapidxDni]=sort(mapDni,2);

for i=1:m%��ǰm�������ļ�Ȩ���ڰ�������
    %��λ������
    if all_label(i,1)==1
        idx_inDi=find(Di_index==i);%��ȡ�ڸ������ݼ��е�λ��
        %�ҵ���������k���ڲ��Ȩ������
        per=Di(mapidxDi(idx_inDi,1:k+1),:);
        per(1,:)=per(1,:).*weight1;
        per(2:end,:)=per(2:end,:).*(weight2/(k));
        new_data=[new_data;per];
        new_label=[new_label;ones(k+1,1)];
    else
        idx_inDni=find(Dni_index==i);%��ȡ�ڸ������ݼ��е�λ��
        %�ҵ���������k���ڲ��Ȩ������
        per=Dni(mapidxDni(idx_inDni,1:k+1),:);
        per(1,:)=per(1,:).*weight1;
        per(2:end,:)=per(2:end,:).*(weight2/(k));
        new_data=[new_data;per];
        new_label=[new_label;zeros(k+1,1)];
    end
end
end