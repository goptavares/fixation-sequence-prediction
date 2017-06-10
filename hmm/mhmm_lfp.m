% Requires the Bayes Net Toolbox for Matlab
% which can be found here: https://github.com/bayesnet/bnt


function model = mhmm_lfp(freq,channel)

lfp             = squeeze(freq.powspctrm(:,channel,:,:));
lfp             = convert_to_cell(lfp);


c               = cvpartition(length(lfp),'HoldOut',0.1);
idx_training    = find(c.training);
idx_testing     = find(c.test);

% Set up parameters 
O        = size(lfp{1},1); %Number of coefficients in a vector 
M        = 2;           %Number of mixtures 
Q        = 2;           %Number of states 
cov_type = 'full';

% initial guess of parameters
prior0        = normalise(rand(Q,1));
transmat0     = mk_stochastic(rand(Q,Q));
[mu0, Sigma0] = mixgauss_init(Q*M, cell2mat(lfp(idx_training)), cov_type);
mu0           = reshape(mu0, [O Q M]);
Sigma0        = reshape(Sigma0, [O O Q M]);
mixmat0       = mk_stochastic(rand(Q,M));

[log_l, prior1, transmat1, mu1, Sigma1, mixmat1] = ...
mhmm_em(lfp(idx_training), prior0, transmat0,...
mu0, Sigma0, mixmat0, 'max_iter', 1000,'thresh',  1e-6);    



[loglik,sig,null]   = get_loglik_neural(lfp(idx_testing),prior1,...
                      transmat1,mu1,Sigma1,mixmat1);

model            = struct();
model.log_l      = log_l;
model.prior_t    = prior1;
model.transmat_t = transmat1;
model.mu1        = mu1;
model.Sigma1     = Sigma1;
model.mixmat1    = mixmat1;
model.loglik     = loglik;
model.sig        = sig;
model.null       = null;


end % Main Function


% 5. Compute a metric for how well the neural model fits
function [loglik,sig,shuffled]   = get_loglik_neural(data,prior,trans,mu1,...
                          Sigma1,mixmat1)

loglik = mhmm_logprob(data, prior, trans,mu1, Sigma1, mixmat1);
nr_iter  = 1000;
shuffled = NaN(1,nr_iter);

for i=1:nr_iter
    % Permute data
    data = cellfun(@(x) Shuffle(x,1),data,'UniformOutput',false);
    shuffled(i) = mhmm_logprob(data, prior,...
                  trans,mu1, Sigma1, mixmat1);
end
                          
sig = 1-sum(loglik>shuffled)./nr_iter;  

end



function lfp    = convert_to_cell(lfp)

temp = cell(1,size(lfp,1));
for i=1:size(lfp,1)
    
   temp{i} = squeeze(lfp(i,:,:)); 
end
lfp = temp;

end
