% Write eye tracking data to file in order to process elsewhere

function write_eye_tracking_to_txt(filename,data, time_step)


to_write = cell(1,length(data));

for i=1:length(data)
    to_write{i} = synchronous_behavior(data(i),time_step);
    to_write{i} = [i*ones(size(to_write{i},1),1),to_write{i}];
end

dlmwrite(filename,cell2mat(to_write'),'newline', 'pc');

end % Main Function



% Convert from asynchronous sampling of behavior to synchronous
function data = synchronous_behavior(behavior,time_step)


nr_trials   = behavior.trialsUsed;
start_stop  = behavior.startStop;
null_vector = zeros(1,3);
data        = [];
novelty     = getNrFixOnStim(start_stop);


start_stop  = [start_stop,novelty];



for i= 1:length(nr_trials)
   
    % Fixations this trial
    idx_trial   = ismember(start_stop(:,4),nr_trials(i));
    trial_fix   = start_stop(idx_trial,:);
    fix_times   = num2cell(trial_fix(:,1:2)',1);

    % Where to sample
    start_trial = behavior.trial_mark(i,1);
    end_trial   = behavior.trial_mark(i,2);
    ts          = start_trial:(time_step*1e6):end_trial;
    
    for j=1:length(ts)
        which_fix = find(cellfun(@(x) ts(j)>=x(1) & ts(j)<=x(2),fix_times));
        if ~isempty(which_fix)
            
            category = find(cellfun(@(x) ...
                       ismember(trial_fix(which_fix,5),x),...
                       behavior.categoryPool));
       
            to_add = [nr_trials(i),...
                      trial_fix(which_fix,3),...
                      category,...
                      trial_fix(which_fix,11)];   
                  
            data   = cat(1,data,to_add);
        else
            data   = cat(1,data,[nr_trials(i) null_vector]);
        end
       
    end
end

end




