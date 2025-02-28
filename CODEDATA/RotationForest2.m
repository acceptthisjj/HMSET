function [X_new ,Y_new , Ri_order] = RotationForest2(X,Y,K)
%XΪ��N * n��ѵ�������ݣ�YΪ��Ӧ��ǩ(N * 1)��KΪ���ֵ������Ӽ�����

N = size(X,1);%ѵ��������
n = size(X,2);%��������
M = floor(n/K);%ÿ�������Ӽ���������������
classnum = size(unique(Y),1);%�����
class = unique(Y)';
subMethod = 3;%��ȡ�Ӽ�����1������������ȡ��%��ȡ�Ӽ�����2��ֱ�������ȡ 
%��ȡ�Ӽ�����3������޳�������������Ȼ����������ȡ

if(rem(n,K) ~= 0)%��������Ӽ����ܱ����֣���ʣ��ķ�Ϊһ�������Ӽ�
    K = K+1;
end

% rand("seed",seed);
F_sub = {};%�����Ӽ�

randcol = randperm(n,n);
%�����ȡ�����Ӽ�
for i = 1:K
    if(i == K)
         F_sub{i,1} = randcol(1,(1+(i-1)*M:end));
        break;
    end
    F_sub{i,1} = randcol(1,(1+(i-1)*M:i*M));%��������Ӽ�
end

%�����ȡ75%������������Ri����
boostrap = 0.75;%������������
subRi=cell(1,K);
Ri = zeros(n,n);

for i = 1:K
     %��ȡ�Ӽ�����1������������ȡ��������
     if(subMethod == 1)
         subRow = [];
         subClassNum = ceil(classnum/2);
         subClass = class(:,randperm(classnum,subClassNum));
         for ic = subClass
             subRow1 = find(Y == ic);
             subRow1 = subRow1(:,1:ceil(size(subRow1,2)*boostrap));
             subRow = [subRow subRow1];
         end
         randRow = randperm(size(subRow,2),size(subRow,2));
         Row = subRow(:,randRow);
         subTrain = X(Row,F_sub{i,1}); 
     elseif(subMethod == 2)%��ȡ�Ӽ�����2��ֱ�������ȡ
         Row = randperm(N,round(N*boostrap));
         subTrain = X(Row,F_sub{i,1});
     elseif(subMethod == 3)
         subRow = [];
%          subClassNum = ceil(classnum/2);%ȡ����Ӽ�
         subClassNum = ceil(classnum/1);%���޳�
%          if(N<150)
%             subClassNum = classnum;%ȫ�����
%          end
         subClass = class(:,randperm(classnum,subClassNum));%ȡһ������Ӽ�
         for ic = subClass
             subRow1 = find(Y == ic);
             subRow1 = subRow1(1:ceil(size(subRow1,1)*boostrap),:);
             subRow = [subRow;subRow1];
         end
         randRow = randperm(size(subRow,1),size(subRow,1));
         Row = subRow(randRow,:)';
         subTrain = X(Row,F_sub{i,1}); 
     end
 
     [U,~] = pca(subTrain);
     subRi{1,i} = U;
     if(i == K)
         Ri((1+(i-1)*M):end,(1+(i-1)*M):end) = subRi{1,i};
        break;
    end
     Ri((1+(i-1)*M):(i*M),(1+(i-1)*M):(i*M)) = subRi{1,i};
end

%��Ri���������н�������
Ri = [Ri;randcol];
Ri_order = sortrows(Ri',size(Ri,1))';
Ri_order = Ri_order(1:end-1,:);

X_new = X*Ri_order;
Y_new = Y;
end