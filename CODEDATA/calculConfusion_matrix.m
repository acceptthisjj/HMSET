function [a,b,c,d]=calculConfusion_matrix(matrix)
%��Ԥ�� ����ʵ
[m,n]=size(matrix);%������

    for i=1:m
        temp=matrix;
        TP(i)=temp(i,i);
        temp1=matrix(i,:);%ȡһ��
        temp1(i)=0;
        FP(i)=sum(temp1);
        temp2=matrix(:,i);
        temp2(i)=0;
        FN(i)=sum(temp1);
        temp3=matrix;
        temp3(i,:)=0;
        temp3(:,i)=0;
        TN(i)=sum(sum(temp3));
    end
    a=sum(TP);
    b=sum(FP);
    c=sum(FN);
    d=sum(TN);
end