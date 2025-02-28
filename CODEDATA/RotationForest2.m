function [X_new ,Y_new , Ri_order] = RotationForest2(X,Y,K)
%X为（N * n）训练集数据，Y为对应标签(N * 1)，K为划分的特征子集个数

N = size(X,1);%训练集行数
n = size(X,2);%特征数量
M = floor(n/K);%每个特征子集包含的特征个数
classnum = size(unique(Y),1);%类别数
class = unique(Y)';
subMethod = 3;%抽取子集方法1：按类别随机抽取；%抽取子集方法2：直接随机抽取 
%抽取子集方法3：随机剔除部分类别的数，然后按类别随机抽取

if(rem(n,K) ~= 0)%如果特征子集不能被均分，则剩余的分为一个特征子集
    K = K+1;
end

% rand("seed",seed);
F_sub = {};%特征子集

randcol = randperm(n,n);
%随机抽取特征子集
for i = 1:K
    if(i == K)
         F_sub{i,1} = randcol(1,(1+(i-1)*M:end));
        break;
    end
    F_sub{i,1} = randcol(1,(1+(i-1)*M:i*M));%存放特征子集
end

%随机抽取75%的样本，构造Ri矩阵
boostrap = 0.75;%样本采样比例
subRi=cell(1,K);
Ri = zeros(n,n);

for i = 1:K
     %抽取子集方法1：按类别随机抽取，二分类
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
     elseif(subMethod == 2)%抽取子集方法2：直接随机抽取
         Row = randperm(N,round(N*boostrap));
         subTrain = X(Row,F_sub{i,1});
     elseif(subMethod == 3)
         subRow = [];
%          subClassNum = ceil(classnum/2);%取类别子集
         subClassNum = ceil(classnum/1);%不剔除
%          if(N<150)
%             subClassNum = classnum;%全部类别
%          end
         subClass = class(:,randperm(classnum,subClassNum));%取一半类别子集
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

%对Ri矩阵特征列进行重排
Ri = [Ri;randcol];
Ri_order = sortrows(Ri',size(Ri,1))';
Ri_order = Ri_order(1:end-1,:);

X_new = X*Ri_order;
Y_new = Y;
end