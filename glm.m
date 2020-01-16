pairing = 'stable'; % 'stable' or 'relaxed'
seed = NaN; % To remove seed, set to NaN

if ~isnan(seed)
    input = readtable(strcat('files/glm.', pairing, ...
        '.seed', num2str(seed), '.csv'));
else
    input = readtable(strcat('files/glm.', pairing, '.csv'));
end

vars = input.Properties.VariableNames;

radius_range = flip([0.55 , 0.525, 0.5  , 0.475, 0.45 , ...
    0.425, 0.4  , 0.375, 0.35]);

fprintf('Output for %s pairing with with seed %d\n', pairing, seed)
% selecting neighborhood radius by position in the range (from 1 to 9)
for radius_choice=1:9 
    % features selected: density and frequency growth at a given radius
    % variable to predict: IsNeologism
    radius_selected = [11-radius_choice, 20-radius_choice, 20];
    input_w_intercept = input(:, [11-radius_choice, 20-radius_choice]);
    input_w_intercept.Intercept = ones(size(input, 1), 1);
    nonnan_spearman = ~isnan(input_w_intercept{:, 2});
    % collinearity test
%     [sValue,condIdx,VarDecomp] = ... 
%         collintest(input_w_intercept(nonnan_spearman, :));
    mdl = fitglm(input(:, radius_selected), ...
        'Distribution', 'binomial', 'ResponseVar','IsNeologism');
    fprintf('%s %g\n\n', 'R^2 =', mdl.Rsquared.Adjusted);
    
    var_names = mdl.Coefficients.Properties.RowNames;
 
%     fprintf('sValue\tcondIdx\t%s\t%s\t%s\n', var_names{2}, var_names{3}, ...
%         var_names{1})
%     for i = 1:size(sValue)
%         fprintf('%g\t%g\t%g\t%g\t%g\n',  sValue(i), condIdx(i), ... 
%         VarDecomp(i, 1), VarDecomp(i, 2), VarDecomp(i, 3))
%     end
    
    fprintf('\n')
    
    var_names = mdl.Coefficients.Properties.RowNames;
    for var_num = 1:size(mdl.Coefficients)
        fprintf('%s\t%g\t%g\t%g\t%g\n',  var_names{var_num}, ... 
        mdl.Coefficients.Estimate(var_num), ...
        mdl.Coefficients.SE(var_num), ...
        mdl.Coefficients.tStat(var_num), ...
        mdl.Coefficients.pValue(var_num))
    end
    
    fprintf('\n')
    
end
