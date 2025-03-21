function U = featureExtract2(X,Y,method,type_num,kk)

if nargin <4 
   error('parameters are not enough!');
end

if (~exist('method','var'))
   method = [];
end

if ~isfield(method,'mode')
method.NeighborMode = 'pca';
end

switch method.mode
    case 'pca'
        if ~isfield(method,'K')
        method.K = 2;
        end
        U = pca(X);
    case'lpp'
        if ~isfield(method,'weightmode')
        method.weightmode = 'Binary';
        end
        if ~isfield(method,'knn_k')
        method.knn_k = 5;
        end
        if ~isfield(method,'t')
        method.t = 5;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        A = constructA(X,options);
        U = lpp(X,A);
    case 'lda'
        U = lda(X,Y,type_num);
    case 'lpdp'
        if ~isfield(method,'t')
        method.t = 10;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        A = constructA(X,options); 
        U = lpdp(X,Y,A,method.mu,type_num);
        case 'ldpp'
        if ~isfield(method,'t')
        method.t = 10;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        A = constructA(X,options); 
        U = ldpp(X,Y,A,method.mu,method.gamma,type_num,method.ratio_b,method.ratio_w);
        case 'ldpp_u'
        if ~isfield(method,'t')
        method.t = 10;
        end
        options = [];
        options.k = method.knn_k;
        options.WeightMode = method.weightmode;
        options.t = method.t;
        %调用constructA()函数，由欧式距离矩阵导出LPP算法邻接图 矩阵A，也是权重矩阵W，在一般的降维算法中，直接乘以该权重矩阵，就可以得出映射后的数据U。
        A = constructA(X,options); 
        %调用ldpp_u()函数 
        U = ldpp_u(X,Y,A,method.mu,method.gamma,type_num,method.ratio_b,method.ratio_w,method.M,method.labda2);
    otherwise
        error('mode does not exist!');
end      
end