function DNNR_Adam_MultiRun(Data, ActFun_L1, D_ActFun_L1,ActFun, D_ActFun,...
    alpha_,L_Reg_, lamda_, Nr_, LossFun_, LossFunPar_, DataName_)     
% Nhat-Duc Hoang
    if nargin < 12
        Data = csvread('Dataset1.csv');
        ActFun_L1 = @Sigmoid;
        D_ActFun_L1 = @D_Sigmoid; 
        ActFun = @Sigmoid;
        D_ActFun = @D_Sigmoid; 
        alpha_ = 0.03;
        lamda_ = 0.001;
        Nr_ = [10 10];
        L_Reg_ = 2;       
        LossFun_ = 'MSE_Loss'; 
        LossFunPar_ = 1;
        DataName_ = 'Dataset1';
    end

    S_Time = cputime();      
    RunCase = 'RealRun';   % 'Demo' or  'RealRun'   
    
    [DataZ,mu,sigma] = zscore(Data); 
    
    NRun = 2; % Number of independent runs
    Max_ep = 1000; % Number of training epochs
    
    if strcmp(RunCase, 'Demo')
        NRun = 2; 
        Max_ep = 50;
    end
    
    TestRatio = 0.1;
    L_reg = L_Reg_; % Regularization type: 1 = L1 and 2 = L2  
    lamda = lamda_; % Regularization parameter   
    Nr = Nr_; % Number of neurons in hidden layers    
    alpha = alpha_; % Adam parameter                      
    BatchSize = 16;     
    %%       
    NIdx = 3; % Number of performance indices 
    Ytr_all = [];
    Yte_all = [];
    Ttr_all = [];
    Tte_all = [];     
    
    Performance_tr = zeros(NRun, NIdx);
    Performance_te = zeros(NRun, NIdx);
    TrackLoss_all = zeros(Max_ep, NRun);
    for r = 1 : NRun
        fprintf('r = %d.\n', r);
        [Xtr_z, Ttr_z, Xte_z, Tte_z] = ...
                DataSubSamplingForReg(DataZ,...
                TestRatio);                                    
   
        ActFun_L1Name = func2str(ActFun_L1);  
        ActFunName = func2str(ActFun);              
                       
        LossFun = LossFun_; 
        LossFunPar = LossFunPar_;
        
        SaveLoc = ['MR_Adam_' DataName_ '_MaxEp' num2str(Max_ep) '_' ActFun_L1Name '_' ...
                ActFunName '_L' num2str(L_reg) '_' LossFun '_' num2str(LossFunPar)]; 
         mkdir(SaveLoc);

        [W, TrackLoss_r] = DNNR_Train_Adam(Xtr_z,Ttr_z,Nr,Max_ep, ActFun_L1, D_ActFun_L1,...
            ActFun, D_ActFun,...
            lamda,alpha,L_reg,BatchSize, LossFun, LossFunPar);   
        TrackLoss_all(:, r) = TrackLoss_r;

        save TrainedAdamDNN_ASEL_u5.mat W;
        
        Ytr_z = DNNR_Predict(W,ActFun_L1, ActFun, Xtr_z);    
        Yte_z = DNNR_Predict(W,ActFun_L1, ActFun, Xte_z); 

        Ytr = Ytr_z*sigma(end) + mu(end);
        Yte = Yte_z*sigma(end) + mu(end);
        Ttr = Ttr_z*sigma(end) + mu(end);
        Tte = Tte_z*sigma(end) + mu(end);

        Ytr_all = [Ytr_all; Ytr];
        Yte_all = [Yte_all; Yte];
        Ttr_all = [Ttr_all; Ttr];
        Tte_all = [Tte_all; Tte];

        Output_tr_r = [Ytr Ttr];
        csvwrite([SaveLoc '/Output_tr_' num2str(r) '.csv'], Output_tr_r);

        Output_te_r = [Yte Tte];
        csvwrite([SaveLoc '/Output_te_' num2str(r) '.csv'], Output_te_r);

        RMSEtr = RMSE_Calculation(Ttr, Ytr);
        MAPEtr = MAPE_Calculation(Ttr, Ytr);
        R2tr = R2_Calculation(Ttr, Ytr);

        RMSEte = RMSE_Calculation(Tte, Yte);
        MAPEte = MAPE_Calculation(Tte, Yte);
        R2te = R2_Calculation(Tte, Yte); 

        Performance_tr(r,:) = [RMSEtr MAPEtr R2tr];
        Performance_te(r,:) = [RMSEte MAPEte R2te];
    end % for r = 1 : NRun 

    csvwrite([SaveLoc '/TrainingOutput_All.csv'], [Ytr_all Ttr_all])
    csvwrite([SaveLoc '/TestingOutput_All.csv'], [Yte_all Tte_all]) 

    csvwrite([SaveLoc '/Performance_tr.csv'], Performance_tr);
    csvwrite([SaveLoc '/Performance_te.csv'], Performance_te);
    Statistis_Performance = [mean(Performance_tr)' std(Performance_tr)'...
                             mean(Performance_te)' std(Performance_te)'];
    csvwrite([SaveLoc '/Statistis_Performance.csv'], Statistis_Performance); 
    
    csvwrite([SaveLoc '/TrackLoss_all.csv'], TrackLoss_all); 
    
    AverageTrackLoss = mean(TrackLoss_all, 2);
    csvwrite([SaveLoc '/AverageTrackLoss.csv'], AverageTrackLoss);
    
    E_Time = cputime();
    C_Time = E_Time - S_Time;
    csvwrite([SaveLoc '/TotalRunTime.csv'], C_Time);
end

function [Best_W, TrackingLossFunVal] = DNNR_Train_Adam(X,T,Nr,Max_ep, ActFun_L1,...
                                                        D_ActFun_L1, ActFun,...
                                                        D_ActFun, lamda,alpha,L_reg,...
                                                        BatchSize, LossFun, LossFunPar)
    N = size(X, 1);
    D = size(X, 2);
    NL = length(Nr); % Number of hidden layers    
    W = cell(1,NL+1);
    W{1} = randn(Nr(1),D+1);
    W{NL+1} = randn(1,Nr(NL)+1);
    for i = 2 : NL
        W{i} = randn(Nr(i),Nr(i-1)+1);
    end   
    
    %%
    Best_W = cell(1,NL+1);
    Best_W{1} = randn(Nr(1),D+1);
    Best_W{NL+1} = randn(1,Nr(NL)+1);
    for i = 2 : NL
        Best_W{i} = randn(Nr(i),Nr(i-1)+1);
    end 
    %%
    Best_Loss = 1e10;  
    
    v_gW = cell(1,NL+1);
    v_gW{1} = zeros(Nr(1),D+1);
    v_gW{NL+1} = zeros(1,Nr(NL)+1);    
    for i = 2 : NL
        v_gW{i} = zeros(Nr(i),Nr(i-1)+1);
    end 
    
    s_gW = cell(1,NL+1);
    s_gW{1} = zeros(Nr(1),D+1);
    s_gW{NL+1} = zeros(1,Nr(NL)+1);
    for i = 2 : NL
        s_gW{i} = zeros(Nr(i),Nr(i-1)+1);
    end
    
    gW = cell(1,NL+1);    
    
    gama_v = 0.9; gama_s = 0.999; eps_ = 1e-8;     

    [sBC, eBC] = SetupBatches(X, BatchSize);    
    
    upd_c = ones(1,NL+1); % weight update count
    
    TrackingLossFunVal = zeros(Max_ep,1);
    
    for ep = 1 : Max_ep
        if rem(ep,100) == 0
            fprintf('ep = %d.\n', ep);
        end
        
        randomIdx = randperm(N);
        Xr = X(randomIdx,:);
        Tr = T(randomIdx);    
        BatchNum = round(N/BatchSize); % BatchNumber 
        X_Batch = cell(1, BatchNum);
        T_Batch = cell(1, BatchNum);
        for i = 1 : BatchNum
            s_idx_b_i = sBC(i);
            e_idx_b_i = eBC(i);
            X_Batch{i} = Xr(s_idx_b_i:e_idx_b_i,:);
            T_Batch{i} = Tr(s_idx_b_i:e_idx_b_i);
        end  
        
        for b_idx = 1 : BatchNum
            X_Batch_i = X_Batch{b_idx};
            T_Batch_i = T_Batch{b_idx};
            NiB = size(X_Batch_i,1); % Number of data in the current batch
            
            gW_Sum = cell(1,NL+1); 
            gW_Sum{1} = zeros(Nr(1),D+1);
            gW_Sum{NL+1} = zeros(1,Nr(NL)+1);            
            for i = 2 : NL
                 gW_Sum{i} = zeros(Nr(i),Nr(i-1)+1);
            end  
            
            for k = 1 : NiB
                x_nb = X_Batch_i(k,:)';
                x = [1; x_nb];            
                t = T_Batch_i(k);

                v1 = W{1}*x;            
%                 y1_nb = ActFun(v1); 
                y1_nb = ActFun_L1(v1);

                y1 = [1; y1_nb];  

                y = cell(1, NL);
                y{1} = y1; 

                v_h = cell(1,NL);             
                v_h{1} = [1; v1];         
                for m = 2 : NL % 2,3                
                    v_m = W{m}*y{m-1};                
                    ym_nb = ActFun(v_m);
                    v_h{m} = [1; v_m];      
                    y{m} = [1; ym_nb]; 
                end  

                v = W{NL+1}*y{NL}; 

                y_p = v;

                % MSE_Loss
                % L = 1*(t-y_p)^2 if t-y_p >= 0
                % L = a*(t-y_p)^2 if t-y_p < 0
                if strcmp(LossFun, 'MSE_Loss')  == true
                    % There is a negative sign because error = t - y
                    % dE/dy --> -
                    if (t-y_p) >= 0
                        e = -(t - y_p);
                    else
                        e = -LossFunPar*(t - y_p);
                    end
                end

                if strcmp(LossFun, 'L_Exp_Loss')  == true                    
                    e = (1/LossFunPar)*(exp(-1*LossFunPar*(t-y_p)) - 1);                    
                end

                delta = e;                

                detal_h = cell(1,NL+1);
                detal_h{NL+1} = delta;            

                for m = NL :-1: 1 % 3 2 1
                    e_m = W{m+1}'*detal_h{m+1};                 
                    if m > 1
                        delta_m_0 = D_ActFun(v_h{m}).*e_m; 
        %                 delta_m_0 = (y{m}.*(1-y{m})).*e_m;    
                        detal_h{m} = delta_m_0(2:end); 
                    end
                    if m == 1
                        delta_m_0 = D_ActFun_L1(v_h{m}).*e_m; 
        %                 delta_m_0 = (y{m}.*(1-y{m})).*e_m;    
                        detal_h{m} = delta_m_0(2:end); 
                    end
                end

                gW{1} = detal_h{1}*x'; % gW{1} = -detal_h{1}*x';
                gW_Sum{1} = gW_Sum{1} + gW{1};
                for m = 2 : NL+1               
                    gW{m} = detal_h{m}*y{m-1}'; % gW{m} = -detal_h{m}*y{m-1}';
                    gW_Sum{m} = gW_Sum{m} + gW{m};                 
                end         
               
            end % for k = 1 : NiB
            
            for u = 1 : NL+1                      
                    gW_Sum{u} = gW_Sum{u}/NiB; 
                    if L_reg == 2  
                        gW_Sum{u} = gW_Sum{u} + lamda*W{u};
                    end
                    if L_reg == 1  
                        gW_Sum{u} = gW_Sum{u} + lamda*sign(W{u});
                    end
                    v_gW{u} = gama_v*v_gW{u} + (1-gama_v)*gW_Sum{u};
                    s_gW{u} = gama_s*s_gW{u} + (1-gama_s)*gW_Sum{u}.*gW_Sum{u};
                    v_gWu_b = v_gW{u}/(1-gama_v^(upd_c(u)));
                    s_gWu_b = s_gW{u}/(1-gama_s^(upd_c(u)));
                    if L_reg == 2                
                        W{u} = W{u} - alpha*(v_gWu_b./(eps_ + sqrt(s_gWu_b)));
                    end
                    if L_reg == 1                
                        W{u} = W{u} - alpha*(v_gWu_b./(eps_ + sqrt(s_gWu_b)));
                    end  
                    upd_c(u) = upd_c(u) + 1;
            end % for u = 1 : NL+1        
            
         end % for b_idx = 1 : BatchNum
         
         Y_ep = DNNR_Predict(W,ActFun_L1, ActFun, X);
         Current_Loss = ComputeTotalLoss(Y_ep, T, LossFun, LossFunPar,...
             L_reg, lamda, W); 
            
         if Current_Loss < Best_Loss
            Best_W = W;
            Best_Loss = Current_Loss;
         end  
         
         TrackingLossFunVal(ep) = Best_Loss; 
    end  
        
end

function TotalLoss = ComputeTotalLoss(Y_ep, T, LossFunName, LossFunPar,...
    RegularizationType, RegularizationPar, W)
    N = length(T);
    SampleLoss = zeros(N,1);  
    
    if strcmp(LossFunName, 'MSE_Loss')  == true
        % There is a negative sign because error = t - y
        % dE/dy --> -
        for i = 1 : N
            if (T(i)-Y_ep(i)) >= 0
                SampleLoss(i) = 0.5*(T(i)-Y_ep(i))^2;
            else
                SampleLoss(i) = 0.5*LossFunPar*(T(i)-Y_ep(i))^2;
            end
        end
    end % if strcmp(LossFunName, 'MSE_Loss')  == true
    
    if strcmp(LossFunName, 'L_Exp_Loss')  == true     
        for i = 1 : N
            SampleLoss(i) = abs((1/LossFunPar)*(exp(-1*LossFunPar*(T(i)-Y_ep(i))) - 1));
        end
    end % if strcmp(LossFunName, 'MSE_Loss')  == true
    
    TotalLoss = sum(SampleLoss);
    % lamda,L_reg
    % W = cell(1,NL+1);
    M = length(W);
    if RegularizationType == 1
        for k = 1 : M
            TotalLoss = TotalLoss + RegularizationPar*sum(abs(W{k}(:)));
        end
    end
    
    if RegularizationType == 2
        for k = 1 : M
            TotalLoss = TotalLoss + RegularizationPar*sqrt(sum((W{k}(:)).^2));
        end
    end   
    
end

function [Xtr, Ytr, Xte, Yte] =  DataSubSamplingForReg(Data, TestRatio)   
    % Subsampling data for Regression
    if nargin < 2
       Data = rand(100, 3); 
       TestRatio = 0.2;
    end
    
    NumData = size(Data,1);
    
    RandIdx = randperm(NumData);
    
    TestIndx = RandIdx(1:round(NumData*TestRatio));
    TrainIndx = RandIdx(1+round(NumData*TestRatio) : end);
    
    TestData = Data(TestIndx, :);
    TrainData = Data(TrainIndx, :);
    
    Xtr = TrainData(:,1:end-1);
    Ytr = TrainData(:, end);
    
    Xte = TestData(:,1:end-1);
    Yte = TestData(:, end);

end

function [sBC, eBC] = SetupBatches(X, BatchSize)
    N = size(X,1);    
    Bs = BatchSize;
    Bn = round(N/Bs);
    k = 1;    
    Binit = 1;
    Bc = [Binit];
    while k < Bn
        Binit = Binit + Bs;
        Bc = [Bc Binit];        
        k = k + 1;
    end
    Bc = [Bc N+1];
    sBC = Bc(1:end-1); % Start index of all batches
    eBC = Bc(2:end)-1; % End index of all batches    
end

function f = DNNR_Predict(W,ActFun_L1, ActFun, X)
    N = size(X,1);
    Y = zeros(N,1);
    NL = size(W,2)-1;
    for k = 1 : N
        x_nb = X(k,:)';
        x = [1; x_nb];           

        v1 = W{1}*x;           
%         y1_nb = ActFun(v1);   
        y1_nb = ActFun_L1(v1); 
        y1 = [1; y1_nb];  

        y = cell(1, NL);
        y{1} = y1;

        for m = 2 : NL % 2,3                
            v_m = W{m}*y{m-1};
            ym_nb = ActFun(v_m);
            y{m} = [1; ym_nb]; 
        end           

        v = W{NL+1}*y{NL}; 

        y_p = v;   
        
        Y(k) = y_p;
    end
    f = Y;
end

function f = TanH(x)
    f = (exp(x)-exp(-x))./ (exp(x)+exp(-x));
end

function f = D_TanH(x)
    f = 1-TanH(x).^2;
end

function f = Sigmoid(x)
    f = 1./ (1 + exp(-x));
end

function f = D_Sigmoid(x)
    f = Sigmoid(x).*(1-Sigmoid(x));
end

function f = ReLU(x)
    f = max(0,x);
end

function f = D_ReLU(x)
    f = zeros(size(x));
    f(x>0) = 1;    
end

function f = LeakyReLU(x)
    f = zeros(size(x));
    f(x>0) = x(x>0);
    f(x<=0) = 0.1*x(x<=0);
end

function f = D_LeakyReLU(x)
    f = zeros(size(x));
    f(x>0) = 1;
    f(x<=0) = 0.1;
end

function R2 = R2_Calculation(Ya,Yp)
    % Computing Coefficient of Determination 
    if nargin < 2
       Ya = rand(100,1)*100;
       Yp = Ya + 5*randn(100,1);
    end
    
    Yte = Ya;
    Ytep = Yp;
    
    Ym = mean(Yte);
    SStot = sum((Yte - Ym).^2);
    SSres = sum((Yte - Ytep).^2);
    R2 = 1 - SSres/SStot;
end % function R2 = ComputingR2(Ya,Yp)


function f = RMSE_Calculation(Ya, Yp)
    % Root Mean Squared Error
    if nargin < 2
       Ya = rand(10,1)*100;
       Yp = rand(10,1)*100;
    end    
    f = sqrt(sum((Ya-Yp).^2)/length(Ya));
end % function f = RMSE_Calculation(Ya, Yp)

function f = MAPE_Calculation(x1, x2)
    % Mean absolute percentage error
    if nargin < 2
        x1 = rand(100,1);
        x1(14) = 0;
        x2 = x1 + rand(100,1)*0.1;  
        x2(14) = 1e-5;
    end
    
    if any(x1==0) || any(x2==0)
        disp('Warning some x1 or x2 == 0');
    end
    
    x1(x1==0) = 1e-5;
    x2(x2==0) = 1e-5;
    
    f = 100*(sum(abs((x1 - x2)./x1))/length(x1));
end