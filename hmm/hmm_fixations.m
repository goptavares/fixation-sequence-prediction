% Requires the Bayes Net Toolbox for Matlab
% which can be found here: https://github.com/bayesnet/bnt


function models = hmm_fixations(startstop,allspikes)

% Organize everything into trials
by_trial         = bin_by_trial(startstop, allspikes);

% Behavioral model of fixations
behavioral_model = fit_behavior_fix(by_trial);

% Neural model of fixations
neural_model     = fit_neural_model(by_trial);

% Save the models
models.behavioral           = behavioral_model;
models.neural               = neural_model;


end % End of Main function


%----------------- Helper functions ----------------------%


% 1. Fit behavioral model
function behavioral_model = fit_behavior_fix(by_trial)


nr_states       = 2; % face vs. non face
nr_obs          = 4; % there are four possible categories
prior           = normalise(rand(nr_states,1));
transmat        = mk_stochastic(rand(nr_states,nr_states));
obsmat          = mk_stochastic(rand(nr_states,nr_obs));

idx_training    = 1:2:length(by_trial);
idx_testing     = 2:2:length(by_trial);

[log_l, prior_t, transmat_t, obsmat_t] = dhmm_em(...
                                   {by_trial(idx_training).categories},...
                                   prior,transmat, obsmat,...
                                   'max_iter', 1000,...
                                   'thresh',  1e-6);                             
[loglik,sig,null]                = get_loglik_behavioral(...
                              {by_trial(idx_testing).categories},...
                              prior_t, transmat_t, obsmat_t);                              
behavioral_model.log_l      = log_l;
behavioral_model.prior_t    = prior_t;
behavioral_model.transmat_t = transmat_t;
behavioral_model.obsmat_t   = obsmat_t;
behavioral_model.loglik     = loglik;
behavioral_model.sig        = sig;
behavioral_model.null       = null;

end 

% 2. Fit neural model
function neural_model   = fit_neural_model(by_trial)

spike_seq     = {by_trial.spikes};
nr_cells      = size(spike_seq{1},1);
nr_states     = 2; % Number of mixtures (face vs. non face)
M             = 1;         
cov_type      = 'full';     % Full covariance
idx_training  = 1:2:length(spike_seq);
idx_testing   = 2:2:length(spike_seq);

training_data = cell2mat(spike_seq(idx_training));
testing_data  = cell2mat(spike_seq(idx_testing));


prior0        = normalise(rand(nr_states,1));
transmat0     = mk_stochastic(rand(nr_states,nr_states));
[mu0, Sigma0] = mixgauss_init(nr_states*M, training_data,...
                cov_type);
mu0           = reshape(mu0, [nr_cells nr_states M]);
Sigma0        = reshape(Sigma0, [nr_cells,nr_cells,nr_states,M]);
mixmat0       = mk_stochastic(rand(nr_states,M));


[log_l, prior_t, transmat_t, mu1, Sigma1, mixmat1] = ...
mhmm_em(training_data, prior0, transmat0, mu0, Sigma0,...
mixmat0,'max_iter', 1000,'thresh',  1e-6);    

[loglik,sig,null]   = get_loglik_neural(testing_data,prior_t,transmat_t,mu1,...
                 Sigma1,mixmat1);

neural_model.log_l      = log_l;
neural_model.prior_t    = prior_t;
neural_model.transmat_t = transmat_t;
neural_model.mu1        = mu1;
neural_model.Sigma1     = Sigma1;
neural_model.mixmat1    = mixmat1;
neural_model.loglik     = loglik;
neural_model.sig        = sig;
neural_model.null       = null;





end


% 3. Restructure data by trial
function by_trial       = bin_by_trial(startstop, allspikes)

trials                  =  unique(startstop(:,4));
categories              =  startstop(:,5);
location                =  startstop(:,3);
[lookONFr,lookDuration] = getFRinLookPeriod(startstop,allspikes,false);
spikes                  = cell2mat(lookONFr);

by_trial    =  struct();
for i=1:length(trials)
    
    idx_trial              = ismember(startstop(:,4),trials(i));
    by_trial(i).categories = categories(idx_trial)';
    by_trial(i).location   = location(idx_trial)';
    by_trial(i).lookdur    = lookDuration(idx_trial)';
    by_trial(i).spikes     = spikes(idx_trial,:)';
end

end


% 4. Compute a metric for how well the behavioral model fits
function [loglik,sig,shuffled]   = get_loglik_behavioral(data,prior,trans,obs)

loglik   = dhmm_logprob(data,prior, trans, obs);
nr_iter  = 1000;
shuffled = NaN(1,nr_iter);

for i=1:nr_iter
    % Permute data
    data      = cellfun(@(x) Shuffle(x),data,'UniformOutput',false);
    shuffled(i) = dhmm_logprob(data,prior, trans, obs);
end
                          
sig = 1-sum(loglik>shuffled)./nr_iter;  


end


% 5. Compute a metric for how well the neural model fits
function [loglik,sig,shuffled]   = get_loglik_neural(data,prior,trans,mu1,...
                          Sigma1,mixmat1)

loglik = mhmm_logprob(data, prior, trans,mu1, Sigma1, mixmat1);
nr_iter  = 1000;
shuffled = NaN(1,nr_iter);

for i=1:nr_iter
    % Permute data
    idx_new     = randperm(size(data,2));
    data        = data(:,idx_new);
    shuffled(i) = mhmm_logprob(data, prior,...
                  trans,mu1, Sigma1, mixmat1);
end
                          
sig = 1-sum(loglik>shuffled)./nr_iter;  

end


