function Z = myprojectData(X, U, K)
%
data = bsxfun(@minus,X,mean(X,1));%ÿ�е�ֵ��ȥ������Ӧ�еľ�ֵ
Z = zeros(size(data, 1), K);
Z = data * U(:,1:K);  % ȡU������ǰK ����Ϊӳ����� �� ��������X ������˵õ� ͨ��ldpp�㷨��ά������ݡ� 
end
