function [X_src_new,X_tar_new,A] = JPSCA(X_src,X_tar,options)
% The is the implementation of Transfer Component Analysis.
% Reference: Sinno Pan et al. Domain Adaptation via Transfer Component Analysis. TNN 2011.

% Inputs: 
%%% X_src          :    source feature matrix, ns * n_feature
%%% X_tar          :    target feature matrix, nt * n_feature
%%% options        :    option struct
%%%%% lambda       :    regularization parameter
%%%%% dim          :    dimensionality after adaptation (dim <= n_feature)
%%%%% kernel_tpye  :    kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% gamma        :    bandwidth for rbf kernel, can be missed for other kernels

% Outputs: 
%%% X_src_new      :    transformed source feature matrix, ns * dim
%%% X_tar_new      :    transformed target feature matrix, nt * dim
%%% A              :    adaptation matrix, (ns + nt) * (ns + nt)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%% Set options
	lambda = options.lambda; 
    
	dim = options.dim;                    
	kernel_type = options.kernel_type;    
	gamma = options.gamma;                

	%% Calculate
	X = [X_src',X_tar'];
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	M = e * e';
	M = M / norm(M,'fro');%��Frobenius ����
	H = eye(n)-1/(n)*ones(n,n);
	if strcmp(kernel_type,'primal')
        [L,D] = my_constructA(X',options);%������˹���� L
        %[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
        [A,~] = eigs(X*(M+L)*X'+lambda*eye(m),X*D*X',dim,'SM');%�Ķ�
		Z = A' * X;
        Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
		X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
    else
	    K = TCA_kernel(kernel_type,X,[],gamma);%��ӳ�����K
        [L,D] = my_constructA(X',options);%������˹���� L
%         [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');%TCA,�����TCAresult��
%          [A,~] = eigs(K*(M+L)*K',K*H*K',dim,'SM');%Ŀ��1ʽԼ��1��JPSCA,complex
%         [A,~] = eigs(K*(M+L)*K'+lambda*eye(n),K*H*K',dim,'SM');%Ŀ��1ʽԼ��1+KK,�����JPSCA_kkresult��
% 	    [A,~] = eigs(K*(M+L)*K',K*(H+D)*K',dim,'SM');%Ŀ��1ʽԼ��2�����ָ������뱨��
	    [A,~] = eigs(K*(M+L)*K'+lambda*eye(n),K*D*K',dim,'SM');%Ŀ��1ʽԼ��2+kk,�����JPSCA_Dkkresult��  ʹ������Լ��ʱû�н�
%         [A,~] = eigs(K*(M+L)*K'+lambda*eye(n),K*(H+D)*K',dim,'SM');%Ŀ��1ʽԼ��1+2+KK,�����JPSCA_HDkkresult��
%%%%%%%%%%%%%%%%%%%%%%%%%����Ȩ%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           [A,~] = eigs(K*(miu1*M+miu2*L)*K'+lambda*eye(n),K*H*K',dim,'SM');%miu=1,acc=0.5635
%         [A,~] = eigs(K*(M+miu*L)*K'+lambda*eye(n),K*H*K',dim,'SM');%miu=1,acc=0.5635
%         [A,~] = eigs(K*(M+miu*L)*K'+lambda*eye(n),K*H*K',dim,'SM');%miu=0.9,acc=0.5615
%         [A,~] = eigs(K*((1-miu)*M+miu*L)*K'+lambda*eye(n),K*H*K',dim,'SM');%miu=0.2,acc=0.5615
        Z = A' * K;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	end
end

% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013
function K = TCA_kernel(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end