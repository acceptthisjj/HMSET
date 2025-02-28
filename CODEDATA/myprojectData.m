function Z = myprojectData(X, U, K)
%
data = bsxfun(@minus,X,mean(X,1));%每列的值减去样本对应列的均值
Z = zeros(size(data, 1), K);
Z = data * U(:,1:K);  % 取U向量的前K 个作为映射矩阵 和 数据样本X 进行相乘得到 通过ldpp算法降维后的数据。 
end
