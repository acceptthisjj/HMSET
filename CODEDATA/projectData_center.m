function Z = projectData_center(X, U, K)

Z = zeros(size(X, 1), K);
Z = X * U(:,1:K);  % ȡU������ǰK ����Ϊӳ����� �� ��������X ������˵õ� ͨ��ldpp�㷨��ά������ݡ� 

Z=Z';
Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
Z=Z';
end
