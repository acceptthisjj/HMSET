function Z = projectData(X, U, K)

Z = zeros(size(X, 1), K);
Z = X * U(:,1:K);  % 取U向量的前K 个作为映射矩阵 和 数据样本X 进行相乘得到 通过ldpp算法降维后的数据。 
end
