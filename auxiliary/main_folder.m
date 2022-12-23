% location function
% 
% This function returns the correct folder name depending on whether the 
% code is for the repository of for developer.
% 
% 
% location = main_folder()
% 
% output:   location = string with correct folder name
% 

function location = main_folder()

dir = 'repo';

if strcmp(dir, 'repo')
    location = 'PSOAnalysis-main';
elseif strcmp(dir, 'developer')
    location = 'CBOandPSO';
else
    error()
end

end