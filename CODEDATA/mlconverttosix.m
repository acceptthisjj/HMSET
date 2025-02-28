function [U] = mlconverttosix(X,method,b) 
[m, n] = size(X);
U1 = zeros(6,n);
%��һ�ν�ά����
U1(1,:)=mean(X,1);%���о�ֵ
U1(2,:)=median(X,1);%��λ��
U1(4,:)=std(X,1);% �����׼�ȨֵΪ1��ά��Ϊ1��Ҳ���Ǽ����б�׼��
U1(5,:)=mad(X);%ƽ������ƫ��
U1(6,:)=quantile(X,0.75,1)-quantile(X,0.25,1);%�ķ�λ��Χ
[val,ind]=sort(X);
U1(3,:)=mean(val(1+floor(b*0.2):1+floor(b*0.8),:),1);% 20%��β��ֵ ǰ1 ��1
%�ڶ��μ���
U2 = zeros(6,n);

U2(1,:)=mean(U1,1);%���о�ֵ
U2(2,:)=median(U1,1);%��λ��
U2(4,:)=std(U1,1);% �����׼�ȨֵΪ1��ά��Ϊ1��Ҳ���Ǽ����б�׼��
U2(5,:)=mad(U1);%ƽ������ƫ��
U2(6,:)=quantile(U1,0.75,1)-quantile(U1,0.25,1);%�ķ�λ��Χ
[val2,ind2]=sort(U1);
U2(3,:)=mean(val2(2:5,:),1);% 20%��β��ֵ ǰ1 ��1

switch method.mode
    case'141'
        U=cat(1,U1(1,:),U1(4,:));
    case'251'
        U=cat(1,U1(2,:),U1(5,:));
    case'361'
        U=cat(1,U1(3,:),U1(6,:));
    case'all1'
        U=U1;
    case'142'
        U=cat(1,U2(1,:),U2(4,:));
    case'252'
        U=cat(1,U2(2,:),U2(5,:));
    case'362'
        U=cat(1,U2(3,:),U2(6,:));
    case'all2'
        U=U2;
    case'1'
        U=U1(1,:);
    case'2'
        U=U1(2,:);
    case'3'
        U=U1(3,:); 
    case'4'
        U=U1(4,:);
    case'5'
        U=U1(5,:);
    case'6'
        U=U1(6,:);
end

end