function [UX,UY] = process1(X,Y,type,a,b) 
%�������ݼ� ��ÿ�����˵�6��Ƭ�� ��ά������Ƭ�� ��ֵ����λ������׼�ƽ������ƫ��ķ�λ��Χ����β��ֵ
%a ���� b һ���˵�������

switch type
    case 1
        method.mode='141';
    case 2
        method.mode='251';
    case 3
        method.mode='361';
    case 4
        method.mode='all1';
end

[m, n] = size(X);
dataperone=[];
for i=1:a
    dataper=X(1+b*(i-1):b+b*(i-1),:);%ѡ��һ�����˵�Ƭ��
    dataperone=cat(1,dataperone,mlconverttosix(dataper,method,b));%���н�ά
end
UX=dataperone;

[m, n] = size(Y);
dataperone1=[];
switch method.mode
    case'141'
        s=2;
    case'251'
        s=2;
    case'361'
        s=2;
    case'all1'
        s=6;
end

for i=1:a
    tempY=[];
    for count=1:s
        tempY=[tempY;Y(1+b*(i-1),:)];
    end
    dataper1=tempY;%ѡ��
    dataperone1=cat(1,dataperone1,dataper1);%ƥ��ddrx�ı�ǩ
end
UY=dataperone1;
end