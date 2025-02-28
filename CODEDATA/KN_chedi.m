%����ÿ������ŷʽ���k��������̽Ѱ�㣩��������̲�ؾ���·��
%���ز�ؾ���ͼ
% step1: Calculate the k nearest distance 
function [D]=KN_chedi(X,k2)
[m, ~] = size(X);
D = zeros(m);
%�趨��ֱ��ĸĵ�����㣨�ýڵ�Ŀ��ƶ�������
k1=k2;
for i =1 : m
    xx = repmat(X(i, :), m, 1);
    diff = xx - X;
    dist = sum(diff.* diff, 2);%�����������������������
    [dd, pos] = sort(dist);
    index = pos(1 : k1 + 1);
    index2 = pos(k1 + 2 : m);
    D(i,index) = sqrt(dist(index));
    D(i,index2) = inf;
end
%step2: recalculate shortest distant matrix
for k=1:m
    for i=1:m
        for j=1:m
            if D(i,j)>D(i,k)+D(k,j)
                D(i,j)=D(i,k)+D(k,j);
            end
        end
    end
end
end